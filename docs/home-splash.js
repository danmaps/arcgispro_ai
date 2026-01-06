(() => {
  const canvas = document.getElementById("home-splash");
  if (!canvas) return;

  const rootElement = document.body || document.documentElement;
  const disableSplash =
    canvas.hasAttribute("data-disabled") ||
    (rootElement && rootElement.dataset.disableSplash === "true");

  if (disableSplash) {
    canvas.style.display = "none";
    return;
  }

  const vertexShaderSource = `
    attribute vec2 position;
    void main() {
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentShaderSource = `
    precision mediump float;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform vec2 u_mouse;
    uniform float u_intensity;
    uniform float u_pointer;

    const float SQRT3 = 1.7320508;

    vec2 worldToAxial(vec2 p) {
      float q = (2.0 / 3.0) * p.x;
      float r = (-1.0 / 3.0) * p.x + (1.0 / SQRT3) * p.y;
      return vec2(q, r);
    }

    vec2 axialToWorld(vec2 h) {
      float x = 1.5 * h.x;
      float y = SQRT3 * (h.y + h.x * 0.5);
      return vec2(x, y);
    }

    vec3 cubeRound(vec3 cube) {
      vec3 rounded = floor(cube + 0.5);
      vec3 diff = rounded - cube;
      if (abs(diff.x) > abs(diff.y) && abs(diff.x) > abs(diff.z)) {
        rounded.x = -rounded.y - rounded.z;
      } else if (abs(diff.y) > abs(diff.z)) {
        rounded.y = -rounded.x - rounded.z;
      } else {
        rounded.z = -rounded.x - rounded.y;
      }
      return rounded;
    }

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    float sdHex(vec2 p, float radius) {
      p = abs(p);
      return max(dot(p, vec2(0.8660254, 0.5)), p.x) - radius;
    }

    float axialLine(vec2 local, vec2 dir, float thickness, float radius, float insideMask) {
      vec2 normal = vec2(-dir.y, dir.x);
      float distNormal = abs(dot(local, normal));
      float span = abs(dot(local, dir));
      float cap = 1.0 - smoothstep(radius - 0.08, radius, span);
      float line = 1.0 - smoothstep(thickness * 0.5, thickness, distNormal);
      return line * cap * insideMask;
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / u_resolution.xy;
      vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
      vec2 centered = (uv - 0.5) * aspect;

      float gridScale = 5.0;
      vec2 point = centered * gridScale;

      vec2 axial = worldToAxial(point);
      vec3 cube = vec3(axial.x, axial.y, -axial.x - axial.y);
      vec3 rounded = cubeRound(cube);
      vec2 cell = rounded.xy;

      vec2 center = axialToWorld(cell);
      vec2 local = point - center;

      float time = u_time * 0.4;
      float jitter = hash(cell);
      float breathing = sin(time + jitter * 6.2831) * 0.02;
      float radius = 0.65 + breathing;

      vec2 pointer = (u_mouse - 0.5) * aspect * gridScale;
      float hover = smoothstep(1.6, 0.25, length(point - pointer));

      float hexDist = sdHex(local, radius);
      float inside = 1.0 - smoothstep(0.0, 0.05, hexDist);
      float outline = 1.0 - smoothstep(0.02, 0.05, abs(hexDist));

      float baseThickness = 0.028 + breathing * 0.2;
      float boost = mix(1.0, 1.3, hover * u_pointer);
      float thickness = baseThickness * boost;

      vec2 axis0 = normalize(vec2(1.0, 0.0));
      vec2 axis1 = normalize(vec2(0.5, SQRT3 * 0.5));
      vec2 axis2 = normalize(vec2(-0.5, SQRT3 * 0.5));

      float line0 = axialLine(local, axis0, thickness, radius, inside);
      float line1 = axialLine(local, axis1, thickness, radius, inside);
      float line2 = axialLine(local, axis2, thickness, radius, inside);
      float lines = max(line0, max(line1, line2));

      float grid = max(outline, lines);

      vec3 base = vec3(0.008, 0.04, 0.13);
      vec3 accent = vec3(0.02, 0.08, 0.24);
      float gradient = 0.5 + 0.5 * sin(dot(centered, vec2(2.0, -1.2)) + time * 0.1 + jitter * 6.2831);
      vec3 background = mix(base, accent, gradient * 0.2 + u_intensity * 0.1);
      background += hover * 0.03;

      vec3 color = mix(background, vec3(0.0), grid);
      float vignette = smoothstep(1.45, 0.15, length(centered));
      color *= vignette;
      color = clamp(color, 0.0, 1.0);

      gl_FragColor = vec4(color, mix(0.4, 0.85, grid) * vignette);
    }
  `;

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let teardown = null;

  const start = () => {
    if (teardown) {
      teardown();
      teardown = null;
    }
    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: false,
      powerPreference: "low-power",
    });

    if (!gl) {
      canvas.style.opacity = "0.28";
      return;
    }

    const runner = startSplash(gl);
    if (runner) {
      teardown = runner;
      canvas.style.opacity = "0.65";
    } else {
      canvas.style.opacity = "0.28";
    }
  };

  const stop = () => {
    if (teardown) {
      teardown();
      teardown = null;
    }
    canvas.style.opacity = prefersReducedMotion.matches ? "0.25" : "0.35";
  };

  if (!prefersReducedMotion.matches) {
    start();
  } else {
    canvas.style.opacity = "0.25";
  }

  const handleMotionChange = (event) => {
    if (event.matches) {
      stop();
      canvas.style.opacity = "0.25";
    } else {
      start();
      canvas.style.opacity = "0.65";
    }
  };

  if (typeof prefersReducedMotion.addEventListener === "function") {
    prefersReducedMotion.addEventListener("change", handleMotionChange);
  } else if (typeof prefersReducedMotion.addListener === "function") {
    prefersReducedMotion.addListener(handleMotionChange);
  }

  canvas.addEventListener(
    "webglcontextlost",
    (event) => {
      event.preventDefault();
      stop();
    },
    false,
  );

  canvas.addEventListener(
    "webglcontextrestored",
    () => {
      if (!prefersReducedMotion.matches) {
        start();
      }
    },
    false,
  );

  function startSplash(gl) {
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    if (!vertexShader || !fragmentShader) {
      if (vertexShader) gl.deleteShader(vertexShader);
      if (fragmentShader) gl.deleteShader(fragmentShader);
      return null;
    }

    const program = createProgram(gl, vertexShader, fragmentShader);
    if (!program) {
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      return null;
    }

    const positionLocation = gl.getAttribLocation(program, "position");
    const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
    const timeLocation = gl.getUniformLocation(program, "u_time");
    const mouseLocation = gl.getUniformLocation(program, "u_mouse");
    const intensityLocation = gl.getUniformLocation(program, "u_intensity");
    const pointerLocation = gl.getUniformLocation(program, "u_pointer");

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);

    const pointer = {
      x: 0.5,
      y: 0.5,
      targetX: 0.5,
      targetY: 0.5,
      strength: 0,
      targetStrength: 0,
    };

    const handlePointer = (event) => {
      pointer.targetX = event.clientX / window.innerWidth;
      pointer.targetY = 1 - event.clientY / window.innerHeight;
      pointer.targetStrength = 1;
    };

    const resetPointer = () => {
      pointer.targetX = 0.5;
      pointer.targetY = 0.5;
      pointer.targetStrength = 0;
    };

    window.addEventListener("pointermove", handlePointer, { passive: true });
    window.addEventListener("pointerdown", handlePointer, { passive: true });
    window.addEventListener("pointerleave", resetPointer);
    document.addEventListener("pointerleave", resetPointer);
    window.addEventListener("blur", resetPointer);

    let rafId = null;
    let elapsed = 0;
    let previous = performance.now();

    const resizeCanvas = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.floor(window.innerWidth * dpr);
      const height = Math.floor(window.innerHeight * dpr);
      if (gl.canvas.width !== width || gl.canvas.height !== height) {
        gl.canvas.width = width;
        gl.canvas.height = height;
      }
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const handleVisibility = () => {
      if (document.hidden) {
        if (rafId) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
      } else if (!rafId) {
        previous = performance.now();
        rafId = requestAnimationFrame(drawFrame);
      }
    };

    document.addEventListener("visibilitychange", handleVisibility);

    const drawFrame = (now) => {
      elapsed += now - previous;
      previous = now;

      pointer.x += (pointer.targetX - pointer.x) * 0.08;
      pointer.y += (pointer.targetY - pointer.y) * 0.08;
      pointer.strength += (pointer.targetStrength - pointer.strength) * 0.05;
      pointer.targetStrength *= 0.96;

      resizeCanvas();
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(resolutionLocation, gl.canvas.width, gl.canvas.height);
      gl.uniform1f(timeLocation, elapsed * 0.001);
      gl.uniform2f(mouseLocation, pointer.x, pointer.y);
      gl.uniform1f(intensityLocation, 0.75);
      gl.uniform1f(pointerLocation, pointer.strength);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      rafId = requestAnimationFrame(drawFrame);
    };

    rafId = requestAnimationFrame((now) => {
      previous = now;
      drawFrame(now);
    });

    return () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      window.removeEventListener("pointermove", handlePointer);
      window.removeEventListener("pointerdown", handlePointer);
      window.removeEventListener("pointerleave", resetPointer);
      document.removeEventListener("pointerleave", resetPointer);
      window.removeEventListener("blur", resetPointer);
      window.removeEventListener("resize", resizeCanvas);
      document.removeEventListener("visibilitychange", handleVisibility);
      if (positionBuffer) gl.deleteBuffer(positionBuffer);
      if (program) gl.deleteProgram(program);
      if (vertexShader) gl.deleteShader(vertexShader);
      if (fragmentShader) gl.deleteShader(fragmentShader);
    };
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    if (!program) return null;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn(gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }
    return program;
  }
})();
