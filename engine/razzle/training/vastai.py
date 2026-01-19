"""
Vast.ai integration for cloud GPU training.

Provides utilities to:
- Search for available GPU instances
- Launch training jobs
- Monitor and retrieve results
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import subprocess
import json
import time
from pathlib import Path


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

    def execute(self, instance_id: int, command: str, timeout: int = 3600) -> str:
        """Execute a command on an instance via SSH."""
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
        ]

        # Retry SSH connection a few times
        for attempt in range(5):
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
            if 'Connection refused' in result.stderr:
                time.sleep(10)
                continue
            # Other error - don't retry
            break

        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout

    def copy_to(
        self,
        instance_id: int,
        local_path: Path,
        remote_path: str,
        retries: int = 5,
        retry_delay: int = 10
    ) -> None:
        """Copy file to instance using SCP."""
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError("Instance not ready")

        ssh_opts = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=30',
        ]

        for attempt in range(retries):
            result = subprocess.run(
                ['scp'] + ssh_opts + [
                    '-P', str(instance.ssh_port),
                    str(local_path),
                    f'root@{instance.ssh_host}:{remote_path}'
                ],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return
            if 'Connection refused' in result.stderr or 'Connection closed' in result.stderr:
                time.sleep(retry_delay)
                continue
            # Other error
            break

        raise RuntimeError(f"Failed to copy {local_path}: {result.stderr}")

    def copy_from(
        self,
        instance_id: int,
        remote_path: str,
        local_path: Path,
        retries: int = 5,
        retry_delay: int = 10
    ) -> None:
        """Copy file from instance using SCP."""
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError("Instance not ready")

        ssh_opts = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=30',
        ]

        for attempt in range(retries):
            result = subprocess.run(
                ['scp'] + ssh_opts + [
                    '-P', str(instance.ssh_port),
                    f'root@{instance.ssh_host}:{remote_path}',
                    str(local_path)
                ],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return
            if 'Connection refused' in result.stderr or 'Connection closed' in result.stderr:
                time.sleep(retry_delay)
                continue
            # Other error
            break

        raise RuntimeError(f"Failed to copy {remote_path}: {result.stderr}")


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
