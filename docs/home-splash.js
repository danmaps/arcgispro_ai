(function () {
  const canvas = document.getElementById("hero-splash");
  if (!canvas) {
    return;
  }

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  if (prefersReducedMotion.matches) {
    canvas.style.opacity = "0.2";
    return;
  }

  const gl = canvas.getContext("webgl", { alpha: true, antialias: true });
  if (!gl) {
    canvas.style.opacity = "0.25";
    return;
  }

  const vertexShaderSource = `
    attribute vec2 position;
    varying vec2 vUv;
    void main() {
      vUv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentShaderSource = `
    precision mediump float;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform vec2 u_mouse;
    uniform float u_intensity;

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }

    float fbm(vec2 p) {
      float value = 0.0;
      float amplitude = 0.5;
      float frequency = 1.0;
      for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 1.75;
        amplitude *= 0.5;
      }
      return value;
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / u_resolution.xy;
      vec2 centered = uv - 0.5;
      centered.x *= u_resolution.x / u_resolution.y;
      float time = u_time * 0.12;

      vec2 mouseShift = (u_mouse - 0.5) * vec2(1.7, 1.1);
      float push = exp(-length(centered - mouseShift) * 2.8);

      float layer1 = fbm(centered * 1.2 + vec2(time, -time * 0.6));
      float layer2 = fbm(centered * 0.8 - vec2(time * 0.4, -time * 0.3));
      float layer3 = fbm(centered * 2.2 + vec2(-time * 0.8, time * 0.5));
      float combined = layer1 * 0.5 + layer2 * 0.3 + layer3 * 0.2;
      combined += push * 0.45;
      combined += sin((centered.x + centered.y + time * 0.6) * 3.0) * 0.04;

      vec3 colorA = vec3(0.05, 0.19, 0.38); // Deep blue
      vec3 colorB = vec3(0.01, 0.03, 0.10); // Very dark blue
      vec3 accent = vec3(0.25, 0.65, 0.85); // Bright cyan
      vec3 ember = vec3(0.96, 0.42, 0.35);  // Warm ember
      vec3 color = mix(colorB, colorA, combined);
      color += accent * smoothstep(0.35, 0.85, combined);
      color += ember * smoothstep(0.6, 0.95, combined) * 0.35;

      float vignette = smoothstep(1.35, 0.2, length(centered * 1.1));
      vec3 finalColor = color * (0.6 + 0.5 * combined) * vignette * u_intensity;
      float alpha = vignette * 0.58;

      gl_FragColor = vec4(finalColor, alpha);
    }
  `;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  function createProgram(vertexShader, fragmentShader) {
    if (!vertexShader || !fragmentShader) {
      return null;
    }
    const program = gl.createProgram();
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

  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createProgram(vertexShader, fragmentShader);

  if (!program) {
    canvas.style.opacity = "0.3";
    return;
  }

  const positionLocation = gl.getAttribLocation(program, "position");
  const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
  const timeLocation = gl.getUniformLocation(program, "u_time");
  const mouseLocation = gl.getUniformLocation(program, "u_mouse");
  const intensityLocation = gl.getUniformLocation(program, "u_intensity");

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW
  );

  const pointer = {
    current: { x: 0.5, y: 0.5 },
    target: { x: 0.5, y: 0.5 },
    intensity: 0.75,
    lastMove: performance.now()
  };

  window.addEventListener(
    "pointermove",
    (event) => {
      pointer.target.x = event.clientX / window.innerWidth;
      pointer.target.y = 1 - event.clientY / window.innerHeight;
      pointer.intensity = 0.7 + Math.min(0.3, Math.abs(pointer.target.x - 0.5) + Math.abs(pointer.target.y - 0.5));
      pointer.lastMove = performance.now();
    },
    { passive: true }
  );

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(window.innerWidth * dpr);
    const height = Math.floor(window.innerHeight * dpr);
    if (gl.canvas.width !== width || gl.canvas.height !== height) {
      gl.canvas.width = width;
      gl.canvas.height = height;
    }
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  let rafId = null;
  let startTime = performance.now();

  function draw(now) {
    pointer.current.x += (pointer.target.x - pointer.current.x) * 0.08;
    pointer.current.y += (pointer.target.y - pointer.current.y) * 0.08;

    resizeCanvas();
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2f(resolutionLocation, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(timeLocation, (now - startTime) * 0.001);
    gl.uniform2f(mouseLocation, pointer.current.x, pointer.current.y);

    const idleFactor = Math.min(1, (now - pointer.lastMove) / 4000);
    const easedIntensity = pointer.intensity - idleFactor * 0.25;
    gl.uniform1f(intensityLocation, Math.max(0.55, easedIntensity));

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    rafId = requestAnimationFrame(draw);
  }

  function start() {
    if (rafId == null) {
      startTime = performance.now();
      rafId = requestAnimationFrame(draw);
    }
  }

  function stop() {
    if (rafId != null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stop();
    } else {
      start();
    }
  });

  const handleMotionChange = (event) => {
    if (event.matches) {
      stop();
      canvas.style.opacity = "0.2";
    } else {
      canvas.style.opacity = "0.55";
      start();
    }
  };

  if (typeof prefersReducedMotion.addEventListener === "function") {
    prefersReducedMotion.addEventListener("change", handleMotionChange);
  } else if (typeof prefersReducedMotion.addListener === "function") {
    prefersReducedMotion.addListener(handleMotionChange);
  }

  start();
})();
