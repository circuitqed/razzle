/**
 * WebGL2-accelerated matrix multiplication for neural network inference.
 *
 * Uses fragment shaders to compute C = A @ B on the GPU.
 * Designed as a drop-in replacement for the CPU gemm() in inference.ts.
 *
 * Key design decisions:
 * - Uses R32F textures (1 float per texel) for simplicity
 * - Weight matrices are pre-uploaded as textures (constant across forward passes)
 * - Activation textures are reused (re-uploaded each call)
 * - Output read back via readPixels (synchronous)
 *
 * Requires: WebGL2 + EXT_color_buffer_float
 */

// ---- WebGL2 GEMM Context ----

export class WebGLGemm {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private uALoc: WebGLUniformLocation;
  private uBLoc: WebGLUniformLocation;
  private uKLoc: WebGLUniformLocation;
  private uAHeightLoc: WebGLUniformLocation;
  private uBWidthLoc: WebGLUniformLocation;
  private vao: WebGLVertexArrayObject;

  // Cached textures and framebuffers
  private texCache = new Map<string, WebGLTexture>();
  private fbCache = new Map<string, { fb: WebGLFramebuffer; tex: WebGLTexture }>();

  // Reusable read-back buffer
  private readBuf: Float32Array = new Float32Array(0);

  constructor(canvas?: HTMLCanvasElement | OffscreenCanvas) {
    // Create WebGL2 context
    const cvs = canvas ?? (typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(1, 1)
      : document.createElement('canvas'));

    const gl = cvs.getContext('webgl2', {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;

    if (!gl) throw new Error('WebGL2 not available');

    // Required extension for rendering to float textures
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) throw new Error('EXT_color_buffer_float not available');

    this.gl = gl;

    // Compile shader program
    this.program = this.createProgram();
    this.uALoc = gl.getUniformLocation(this.program, 'uA')!;
    this.uBLoc = gl.getUniformLocation(this.program, 'uB')!;
    this.uKLoc = gl.getUniformLocation(this.program, 'uK')!;
    this.uAHeightLoc = gl.getUniformLocation(this.program, 'uAHeight')!;
    this.uBWidthLoc = gl.getUniformLocation(this.program, 'uBWidth')!;

    // Full-screen quad VAO
    this.vao = this.createQuadVAO();
  }

  /**
   * Upload a weight matrix as a persistent texture.
   * Call once during model initialization for each weight matrix.
   * @param key Unique identifier for this weight matrix
   * @param data Float32Array of the matrix (row-major)
   * @param rows Number of rows (M dimension)
   * @param cols Number of columns (K dimension)
   */
  uploadWeights(key: string, data: Float32Array, rows: number, cols: number): void {
    const existing = this.texCache.get(key);
    if (existing) {
      this.gl.deleteTexture(existing);
    }
    const tex = this.createDataTexture(data, cols, rows);
    this.texCache.set(key, tex);
  }

  /**
   * Compute C = A @ B using the GPU.
   *
   * @param weightKey Key of pre-uploaded weight matrix A (M × K)
   * @param B Activation matrix (K × N), row-major Float32Array
   * @param C Output matrix (M × N), row-major Float32Array (written in-place)
   * @param M Rows of A / rows of C
   * @param K Cols of A / rows of B
   * @param N Cols of B / cols of C
   */
  gemm(weightKey: string, B: Float32Array, C: Float32Array, M: number, K: number, N: number): void {
    const gl = this.gl;
    const aTex = this.texCache.get(weightKey);
    if (!aTex) throw new Error(`Weight texture not found: ${weightKey}`);

    // Upload B as a temporary texture
    const bTex = this.createDataTexture(B, N, K);

    // Get or create framebuffer for output size (M × N)
    const fbKey = `${M}x${N}`;
    let fbEntry = this.fbCache.get(fbKey);
    if (!fbEntry) {
      fbEntry = this.createFramebuffer(N, M);
      this.fbCache.set(fbKey, fbEntry);
    }

    // Bind framebuffer and set viewport
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbEntry.fb);
    gl.viewport(0, 0, N, M);

    // Use program
    gl.useProgram(this.program);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, aTex);
    gl.uniform1i(this.uALoc, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bTex);
    gl.uniform1i(this.uBLoc, 1);

    // Set uniforms
    gl.uniform1i(this.uKLoc, K);
    gl.uniform1i(this.uAHeightLoc, M);
    gl.uniform1i(this.uBWidthLoc, N);

    // Draw full-screen quad
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Read back result
    const totalFloats = M * N * 4; // RGBA
    if (this.readBuf.length < totalFloats) {
      this.readBuf = new Float32Array(totalFloats);
    }
    gl.readPixels(0, 0, N, M, gl.RGBA, gl.FLOAT, this.readBuf);

    // Extract R channel (our result) into C
    for (let i = 0; i < M * N; i++) {
      C[i] = this.readBuf[i * 4];
    }

    // Clean up temporary B texture
    gl.deleteTexture(bTex);

    // Unbind
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Compute C = A @ B where A is provided directly (not pre-uploaded).
   * Used for activation-activation multiplies (e.g., im2col output).
   */
  gemmDirect(
    A: Float32Array, B: Float32Array, C: Float32Array,
    M: number, K: number, N: number,
  ): void {
    const gl = this.gl;

    const aTex = this.createDataTexture(A, K, M);
    const bTex = this.createDataTexture(B, N, K);

    const fbKey = `${M}x${N}`;
    let fbEntry = this.fbCache.get(fbKey);
    if (!fbEntry) {
      fbEntry = this.createFramebuffer(N, M);
      this.fbCache.set(fbKey, fbEntry);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbEntry.fb);
    gl.viewport(0, 0, N, M);
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, aTex);
    gl.uniform1i(this.uALoc, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bTex);
    gl.uniform1i(this.uBLoc, 1);

    gl.uniform1i(this.uKLoc, K);
    gl.uniform1i(this.uAHeightLoc, M);
    gl.uniform1i(this.uBWidthLoc, N);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const totalFloats = M * N * 4;
    if (this.readBuf.length < totalFloats) {
      this.readBuf = new Float32Array(totalFloats);
    }
    gl.readPixels(0, 0, N, M, gl.RGBA, gl.FLOAT, this.readBuf);

    for (let i = 0; i < M * N; i++) {
      C[i] = this.readBuf[i * 4];
    }

    gl.deleteTexture(aTex);
    gl.deleteTexture(bTex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Release all GPU resources.
   */
  dispose(): void {
    const gl = this.gl;
    for (const tex of this.texCache.values()) gl.deleteTexture(tex);
    for (const { fb, tex } of this.fbCache.values()) {
      gl.deleteFramebuffer(fb);
      gl.deleteTexture(tex);
    }
    this.texCache.clear();
    this.fbCache.clear();
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
  }

  // ---- Private helpers ----

  private createProgram(): WebGLProgram {
    const gl = this.gl;

    const vsSource = `#version 300 es
      in vec2 aPosition;
      void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;

    const fsSource = `#version 300 es
      precision highp float;
      precision highp sampler2D;

      uniform sampler2D uA;  // (K cols, M rows) — A[m,k] at texel (k, m)
      uniform sampler2D uB;  // (N cols, K rows) — B[k,n] at texel (n, k)
      uniform int uK;
      uniform int uAHeight;  // M
      uniform int uBWidth;   // N

      out vec4 fragColor;

      void main() {
        int n = int(gl_FragCoord.x);  // column in output
        int m = int(gl_FragCoord.y);  // row in output

        float sum = 0.0;
        for (int k = 0; k < uK; k++) {
          float a = texelFetch(uA, ivec2(k, m), 0).r;
          float b = texelFetch(uB, ivec2(n, k), 0).r;
          sum += a * b;
        }

        fragColor = vec4(sum, 0.0, 0.0, 1.0);
      }
    `;

    const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`Shader link failed: ${info}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile failed: ${info}`);
    }
    return shader;
  }

  private createQuadVAO(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    // Full-screen quad as triangle strip
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(this.program, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
    return vao;
  }

  /**
   * Create a 2D float texture from a row-major Float32Array.
   * Matrix of shape (rows, cols) stored as texture of size (cols, rows).
   * Texel (x, y) = data[y * cols + x].
   */
  private createDataTexture(data: Float32Array, width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);

    // Use R32F: 1 float per texel
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R32F,
      width, height, 0,
      gl.RED, gl.FLOAT, data,
    );

    // No filtering needed (texelFetch uses integer coordinates)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    return tex;
  }

  /**
   * Create a framebuffer with a float color attachment for rendering output.
   */
  private createFramebuffer(width: number, height: number): { fb: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA32F,
      width, height, 0,
      gl.RGBA, gl.FLOAT, null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete: ${status}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fb, tex };
  }
}
