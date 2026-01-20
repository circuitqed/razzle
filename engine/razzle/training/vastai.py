"""
Vast.ai integration for cloud GPU training.

Provides utilities to:
- Search for available GPU instances
- Launch training jobs
- Monitor and retrieve results
- Health checks and robust retry logic
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import subprocess
import json
import time
import hashlib
from pathlib import Path


# Errors that warrant retry (SSH/SCP connection issues)
RETRYABLE_ERRORS = [
    'Connection refused',
    'Connection reset',
    'Connection timed out',
    'Connection closed',
    'No route to host',
    'Network is unreachable',
    'Host is down',
    'Operation timed out',
    'ssh_exchange_identification',
    'kex_exchange_identification',
    'Broken pipe',
]


def is_retryable_error(error_text: str) -> bool:
    """Check if an error message indicates a retryable condition."""
    error_lower = error_text.lower()
    return any(err.lower() in error_lower for err in RETRYABLE_ERRORS)


@dataclass
class GPUOffer:
    """A GPU rental offer from Vast.ai."""
    id: int
    gpu_name: str
    num_gpus: int
    gpu_ram: float  # GB
    cpu_ram: float  # GB
    disk_space: float  # GB
    dph_total: float  # $/hr
    inet_down: float  # Mbps
    inet_up: float  # Mbps
    reliability: float
    dlperf: float  # Deep learning performance score

    @classmethod
    def from_json(cls, data: dict) -> GPUOffer:
        return cls(
            id=data['id'],
            gpu_name=data.get('gpu_name', 'Unknown'),
            num_gpus=data.get('num_gpus', 1),
            gpu_ram=data.get('gpu_ram', 0) / 1024,  # Convert MB to GB
            cpu_ram=data.get('cpu_ram', 0) / 1024,
            disk_space=data.get('disk_space', 0),
            dph_total=data.get('dph_total', 0),
            inet_down=data.get('inet_down', 0),
            inet_up=data.get('inet_up', 0),
            reliability=data.get('reliability2', 0),
            dlperf=data.get('dlperf', 0)
        )


@dataclass
class Instance:
    """A running Vast.ai instance."""
    id: int
    status: str
    ssh_host: Optional[str]
    ssh_port: Optional[int]
    gpu_name: str
    actual_status: str

    @classmethod
    def from_json(cls, data: dict) -> Instance:
        return cls(
            id=data['id'],
            status=data.get('status', 'unknown'),
            ssh_host=data.get('ssh_host'),
            ssh_port=data.get('ssh_port'),
            gpu_name=data.get('gpu_name', 'Unknown'),
            actual_status=data.get('actual_status', 'unknown')
        )


class VastAI:
    """
    Interface to Vast.ai CLI for cloud GPU management.

    Requires: pip install vastai
    And API key configured via: vastai set api-key YOUR_KEY
    """

    def __init__(self):
        self._check_cli()

    def _check_cli(self) -> None:
        """Verify vastai CLI is available."""
        try:
            subprocess.run(['vastai', '--help'], capture_output=True, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                "vastai CLI not found. Install with: pip install vastai\n"
                "Then configure: vastai set api-key YOUR_KEY"
            )

    def _run(self, *args) -> str:
        """Run vastai command and return output."""
        result = subprocess.run(
            ['vastai'] + list(args),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"vastai command failed: {result.stderr}")
        return result.stdout

    def search_offers(
        self,
        gpu_name: Optional[str] = None,
        min_gpu_ram: float = 8,
        max_dph: float = 0.5,
        min_reliability: float = 0.95,
        order_by: str = 'dph_total'
    ) -> list[GPUOffer]:
        """
        Search for available GPU offers.

        Args:
            gpu_name: Filter by GPU name (e.g., 'RTX_3090', 'RTX_4090')
            min_gpu_ram: Minimum GPU RAM in GB (used for post-filtering only,
                         as Vast.ai API doesn't support gpu_ram filter)
            max_dph: Maximum price per hour
            min_reliability: Minimum reliability score
            order_by: Sort field ('dph_total', 'dlperf', etc.)

        Returns list of GPUOffer objects.
        """
        # Note: gpu_ram filter doesn't work in Vast.ai's search API,
        # so we filter by GPU name and post-filter by RAM if needed
        query_parts = [
            f'dph_total<={max_dph}',
            f'reliability2>={min_reliability}',
            'rentable=true'
        ]

        if gpu_name:
            query_parts.append(f'gpu_name={gpu_name}')

        query = ' '.join(query_parts)
        output = self._run('search', 'offers', query, '--raw')

        try:
            data = json.loads(output)
            offers = [GPUOffer.from_json(o) for o in data]
            # Post-filter by GPU RAM since API doesn't support it
            if min_gpu_ram > 0:
                offers = [o for o in offers if o.gpu_ram >= min_gpu_ram]
            # Sort by specified field
            offers.sort(key=lambda o: getattr(o, order_by, 0))
            return offers
        except json.JSONDecodeError:
            return []

    def create_instance(
        self,
        offer_id: int,
        image: str = 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
        disk: int = 20
    ) -> int:
        """
        Create a new instance from an offer.

        Returns instance ID.
        """
        output = self._run(
            'create', 'instance',
            str(offer_id),
            '--image', image,
            '--disk', str(disk),
            '--raw'
        )

        try:
            data = json.loads(output)
            return data.get('new_contract')
        except (json.JSONDecodeError, KeyError):
            raise RuntimeError(f"Failed to create instance: {output}")

    def get_instance(self, instance_id: int) -> Optional[Instance]:
        """Get instance details."""
        output = self._run('show', 'instance', str(instance_id), '--raw')

        try:
            data = json.loads(output)
            if isinstance(data, list) and data:
                return Instance.from_json(data[0])
            elif isinstance(data, dict):
                return Instance.from_json(data)
        except json.JSONDecodeError:
            pass

        return None

    def list_instances(self) -> list[Instance]:
        """List all instances."""
        output = self._run('show', 'instances', '--raw')

        try:
            data = json.loads(output)
            return [Instance.from_json(i) for i in data]
        except json.JSONDecodeError:
            return []

    def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance."""
        self._run('destroy', 'instance', str(instance_id))

    def wait_for_instance(
        self,
        instance_id: int,
        timeout: int = 300,
        poll_interval: int = 10
    ) -> Instance:
        """Wait for instance to be ready (SSH available)."""
        start = time.time()

        while time.time() - start < timeout:
            instance = self.get_instance(instance_id)
            if instance and instance.ssh_host and instance.actual_status == 'running':
                return instance
            time.sleep(poll_interval)

        raise TimeoutError(f"Instance {instance_id} not ready after {timeout}s")

    def execute(
        self,
        instance_id: int,
        command: str,
        timeout: int = 3600,
        retries: int = 5,
        retry_delay: int = 10
    ) -> str:
        """
        Execute a command on an instance via SSH.

        Args:
            instance_id: Vast.ai instance ID
            command: Command to execute
            timeout: Command timeout in seconds
            retries: Number of retry attempts for connection errors
            retry_delay: Delay between retries in seconds

        Returns:
            Command stdout

        Raises:
            RuntimeError: If command fails after all retries
        """
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError("Instance not ready")

        # SSH options for non-interactive execution
        ssh_opts = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=60',
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=10',
            '-o', 'BatchMode=yes',
        ]

        last_error = None

        for attempt in range(retries):
            try:
                result = subprocess.run(
                    ['ssh'] + ssh_opts + [
                        '-p', str(instance.ssh_port),
                        f'root@{instance.ssh_host}',
                        command
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode == 0:
                    return result.stdout

                last_error = result.stderr

                # Check if error is retryable
                if is_retryable_error(result.stderr):
                    time.sleep(retry_delay)
                    continue

                # Non-retryable error
                break

            except subprocess.TimeoutExpired as e:
                last_error = f"Command timed out after {timeout}s"
                # Timeout might be retryable in some cases
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue
                break

        raise RuntimeError(f"Command failed after {retries} attempts: {last_error}")

    def copy_to(
        self,
        instance_id: int,
        local_path: Path,
        remote_path: str,
        retries: int = 5,
        retry_delay: int = 10,
        verify: bool = False
    ) -> None:
        """
        Copy file to instance using SCP.

        Args:
            instance_id: Vast.ai instance ID
            local_path: Local file path
            remote_path: Remote destination path
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            verify: If True, verify file size after transfer

        Raises:
            RuntimeError: If copy fails after all retries
        """
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError("Instance not ready")

        local_path = Path(local_path)
        if not local_path.exists():
            raise RuntimeError(f"Local file not found: {local_path}")

        local_size = local_path.stat().st_size

        ssh_opts = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=30',
            '-o', 'BatchMode=yes',
        ]

        last_error = None

        for attempt in range(retries):
            try:
                result = subprocess.run(
                    ['scp'] + ssh_opts + [
                        '-P', str(instance.ssh_port),
                        str(local_path),
                        f'root@{instance.ssh_host}:{remote_path}'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for large files
                )

                if result.returncode == 0:
                    # Optionally verify file size
                    if verify:
                        try:
                            size_output = self.execute(
                                instance_id,
                                f'stat -c%s {remote_path}',
                                timeout=30,
                                retries=2
                            )
                            remote_size = int(size_output.strip())
                            if remote_size != local_size:
                                last_error = f"Size mismatch: local={local_size}, remote={remote_size}"
                                time.sleep(retry_delay)
                                continue
                        except Exception as e:
                            # Verification failed, but file might be OK
                            pass
                    return

                last_error = result.stderr

                if is_retryable_error(result.stderr):
                    time.sleep(retry_delay)
                    continue

                # Non-retryable error
                break

            except subprocess.TimeoutExpired:
                last_error = "SCP timed out"
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue
                break

        raise RuntimeError(f"Failed to copy {local_path}: {last_error}")

    def copy_from(
        self,
        instance_id: int,
        remote_path: str,
        local_path: Path,
        retries: int = 5,
        retry_delay: int = 10,
        verify: bool = False
    ) -> None:
        """
        Copy file from instance using SCP.

        Args:
            instance_id: Vast.ai instance ID
            remote_path: Remote file path
            local_path: Local destination path
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            verify: If True, verify file size after transfer

        Raises:
            RuntimeError: If copy fails after all retries
        """
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError("Instance not ready")

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Get remote file size for verification
        remote_size = None
        if verify:
            try:
                size_output = self.execute(
                    instance_id,
                    f'stat -c%s {remote_path}',
                    timeout=30,
                    retries=2
                )
                remote_size = int(size_output.strip())
            except Exception:
                pass

        ssh_opts = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=30',
            '-o', 'BatchMode=yes',
        ]

        last_error = None

        for attempt in range(retries):
            try:
                result = subprocess.run(
                    ['scp'] + ssh_opts + [
                        '-P', str(instance.ssh_port),
                        f'root@{instance.ssh_host}:{remote_path}',
                        str(local_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for large files
                )

                if result.returncode == 0:
                    # Verify file size if we know the remote size
                    if verify and remote_size is not None and local_path.exists():
                        local_size = local_path.stat().st_size
                        if local_size != remote_size:
                            last_error = f"Size mismatch: remote={remote_size}, local={local_size}"
                            local_path.unlink()  # Remove incomplete file
                            time.sleep(retry_delay)
                            continue
                    return

                last_error = result.stderr

                if is_retryable_error(result.stderr):
                    time.sleep(retry_delay)
                    continue

                # Non-retryable error
                break

            except subprocess.TimeoutExpired:
                last_error = "SCP timed out"
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue
                break

        raise RuntimeError(f"Failed to copy {remote_path}: {last_error}")

    def check_health(self, instance_id: int) -> dict:
        """
        Check health of an instance.

        Returns dict with:
            - reachable: bool
            - gpu_available: bool
            - gpu_name: str (if available)
            - gpu_memory_mb: int (if available)
            - disk_free_gb: float (if available)
        """
        health = {
            'reachable': False,
            'gpu_available': False,
            'gpu_name': None,
            'gpu_memory_mb': None,
            'disk_free_gb': None,
        }

        try:
            # Basic connectivity check
            result = self.execute(instance_id, 'echo OK', timeout=30, retries=1)
            health['reachable'] = 'OK' in result

            if health['reachable']:
                # Check GPU
                try:
                    gpu_info = self.execute(
                        instance_id,
                        'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits',
                        timeout=30,
                        retries=1
                    )
                    if gpu_info.strip():
                        parts = gpu_info.strip().split(',')
                        health['gpu_available'] = True
                        health['gpu_name'] = parts[0].strip() if len(parts) > 0 else None
                        health['gpu_memory_mb'] = int(parts[1].strip()) if len(parts) > 1 else None
                except Exception:
                    pass

                # Check disk space
                try:
                    disk_info = self.execute(
                        instance_id,
                        "df -BG /workspace | tail -1 | awk '{print $4}'",
                        timeout=30,
                        retries=1
                    )
                    if disk_info.strip():
                        health['disk_free_gb'] = float(disk_info.strip().rstrip('G'))
                except Exception:
                    pass

        except Exception:
            pass

        return health


def find_best_offer(
    target_gpu: str = 'RTX_3090',
    max_price: float = 0.3
) -> Optional[GPUOffer]:
    """
    Find the best value GPU offer.

    Returns the cheapest offer meeting requirements, or None.
    """
    vast = VastAI()
    offers = vast.search_offers(
        gpu_name=target_gpu,
        max_dph=max_price,
        order_by='dph_total'
    )
    return offers[0] if offers else None
