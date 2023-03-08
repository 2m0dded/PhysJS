const canvas = document.getElementById('game-canvas');

const gl = canvas.getContext('webgl');

gl.viewport(0, 0, canvas.width, canvas.height);

gl.clearColor(0.0, 0.0, 0.0, 1.0);
gl.clearDepth(1.0);

gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);

let vertexShader = null;
let fragmentShader = null;
let shaderSource = null;

let vertexBuffer = null;
let indexBuffer = null;

let modelViewMatrix = mat4.create();
let projectionMatrix = mat4.create();
let mvpMatrix = mat4.create();

let positionAttributeLocation = null;
let normalAttributeLocation = null;
let textureAttributeLocation = null;

let texture = null;
let textureLocation = null;

let lightPosition = vec3.create();
let lightColor = vec3.create();
let ambientColor = vec3.create();

let diffuseColor = vec3.create();
let specularColor = vec3.create();
let shininess = 0.0;

let cameraPosition = vec3.create();
let cameraTarget = vec3.create();
let cameraUp = vec3.create();

function createProgram(vertexShaderSource, fragmentShaderSource) {

  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);

  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
    console.error('Error compiling vertex shader:', gl.getShaderInfoLog(vertexShader));
    return null;
  }

  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);

  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
    console.error('Error compiling fragment shader:', gl.getShaderInfoLog(fragmentShader));
    return null;
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Error linking program:', gl.getProgramInfoLog(program));
    return null;
  }

  return program;
}

function createVertexBuffer(data) {

  const buffer = gl.createBuffer();

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  return buffer;
}

function createIndexBuffer(data) {

  const buffer = gl.createBuffer();

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);

  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(data), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

  return buffer;
}

function setVertexAttrib(buffer, location, size, stride, offset) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset);
  gl.enableVertexAttribArray(location);
}

function setUniform(program, name, type, value) {
  const location = gl.getUniformLocation(program, name);
  if (location === null) {
    console.warn(`Uniform '${name}' not found in program`);
    return;
  }
  switch (type) {
    case 'matrix4fv':
      gl.uniformMatrix4fv(location, false, value);
      break;
    case 'vector2fv':
      gl.uniform2fv(location, value);
      break;
    case 'vector3fv':
      gl.uniform3fv(location, value);
      break;
    case 'vector4fv':
      gl.uniform4fv(location, value);
      break;
    case '1f':
      gl.uniform1f(location, value);
      break;
    case '1i':
      gl.uniform1i(location, value);
      break;
    default:
      console.warn(`Unknown uniform type '${type}'`);
  }
}

function createTexture(imageUrl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  };
  image.src = imageUrl;

  return texture;
}

function createFramebuffer(width, height) {
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

  const texture = createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const depthBuffer = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthBuffer);

  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return {
    framebuffer,
    texture,
    depthBuffer
  };
}

function setFramebuffer(framebuffer) {
  if (framebuffer) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer.framebuffer);
    gl.viewport(0, 0, framebuffer.texture.width, framebuffer.texture.height);
  } else {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  }
}

function setTexture(unit, texture) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
}

function createCube(gl, options = {}) {
  const defaults = {
    size: 1,
    xSegments: 1,
    ySegments: 1,
    zSegments: 1,
  };
  const {
    size,
    xSegments,
    ySegments,
    zSegments
  } = Object.assign(defaults, options);

  const x = size / 2;
  const y = size / 2;
  const z = size / 2;

  const positions = [];
  const normals = [];
  const uvs = [];
  const indices = [];

  for (let iz = 0; iz <= zSegments; iz++) {
    const zRatio = iz / zSegments;
    const zPosition = zRatio * size - z;
    for (let iy = 0; iy <= ySegments; iy++) {
      const yRatio = iy / ySegments;
      const yPosition = yRatio * size - y;
      for (let ix = 0; ix <= xSegments; ix++) {
        const xRatio = ix / xSegments;
        const xPosition = xRatio * size - x;

        const cornerVector = [xPosition, yPosition, zPosition];
        positions.push(...cornerVector);

        const normal = vec3.normalize([], cornerVector);
        normals.push(...normal);

        const u = xRatio;
        const v = 1 - yRatio;
        uvs.push(u, v);
      }
    }
  }

  for (let iz = 0; iz < zSegments; iz++) {
    for (let iy = 0; iy < ySegments; iy++) {
      for (let ix = 0; ix < xSegments; ix++) {
        const a = ix + (xSegments + 1) * iy + (xSegments + 1) * (ySegments + 1) * iz;
        const b = ix + (xSegments + 1) * (iy + 1) + (xSegments + 1) * (ySegments + 1) * iz;
        const c = (ix + 1) + (xSegments + 1) * iy + (xSegments + 1) * (ySegments + 1) * iz;
        const d = (ix + 1) + (xSegments + 1) * (iy + 1) + (xSegments + 1) * (ySegments + 1) * iz;
        indices.push(a, b, d);
        indices.push(d, c, a);
      }
    }
  }

  const vertexBuffer = createVertexBuffer(gl, new Float32Array(positions));
  const normalBuffer = createVertexBuffer(gl, new Float32Array(normals));
  const uvBuffer = createVertexBuffer(gl, new Float32Array(uvs));
  const indexBuffer = createIndexBuffer(gl, new Uint16Array(indices));

  return {
    vertexBuffer,
    normalBuffer,
    uvBuffer,
    indexBuffer,
    numVertices: indices.length,
  };
}

function createSphere(radius, latitudeBands, longitudeBands) {
  let vertices = [];
  let normals = [];
  let texCoords = [];
  let indices = [];

  for (let lat = 0; lat <= latitudeBands; lat++) {
    let theta = lat * Math.PI / latitudeBands;
    let sinTheta = Math.sin(theta);
    let cosTheta = Math.cos(theta);

    for (let long = 0; long <= longitudeBands; long++) {
      let phi = long * 2 * Math.PI / longitudeBands;
      let sinPhi = Math.sin(phi);
      let cosPhi = Math.cos(phi);

      let x = cosPhi * sinTheta;
      let y = cosTheta;
      let z = sinPhi * sinTheta;
      let u = 1 - (long / longitudeBands);
      let v = lat / latitudeBands;

      normals.push(x);
      normals.push(y);
      normals.push(z);
      texCoords.push(u);
      texCoords.push(v);
      vertices.push(radius * x);
      vertices.push(radius * y);
      vertices.push(radius * z);
    }
  }

  for (let lat = 0; lat < latitudeBands; lat++) {
    for (let long = 0; long < longitudeBands; long++) {
      let first = (lat * (longitudeBands + 1)) + long;
      let second = first + longitudeBands + 1;
      indices.push(first);
      indices.push(second);
      indices.push(first + 1);

      indices.push(second);
      indices.push(second + 1);
      indices.push(first + 1);
    }
  }

  return {
    positions: vertices,
    normals: normals,
    texCoords: texCoords,
    indices: indices,
  };
}

function createWASDCamera(position, fov, aspect, near, far) {
  const camera = {
    position: position || [0, 0, 0],
    target: [0, 0, -1],
    up: [0, 1, 0],
    fov: fov || 45,
    aspect: aspect || 1,
    near: near || 0.1,
    far: far || 1000,
    speed: 0.1,
  };

  function moveForward() {
    const dir = vec3.normalize([], vec3.subtract([], camera.target, camera.position));
    vec3.scaleAndAdd(camera.position, camera.position, dir, camera.speed);
  }

  function moveBackward() {
    const dir = vec3.normalize([], vec3.subtract([], camera.target, camera.position));
    vec3.scaleAndAdd(camera.position, camera.position, dir, -camera.speed);
  }

  function moveLeft() {
    const right = vec3.normalize([], vec3.cross([], camera.target, camera.up));
    vec3.scaleAndAdd(camera.position, camera.position, right, -camera.speed);
  }

  function moveRight() {
    const right = vec3.normalize([], vec3.cross([], camera.target, camera.up));
    vec3.scaleAndAdd(camera.position, camera.position, right, camera.speed);
  }

  function update() {
    const view = mat4.lookAt([], camera.position, camera.target, camera.up);
    const projection = mat4.perspective([], camera.fov, camera.aspect, camera.near, camera.far);
    const viewProjection = mat4.multiply([], projection, view);
    return {
      view,
      projection,
      viewProjection
    };
  }

  function handleKeyDown(event) {
    switch (event.key) {
      case "w":
        moveForward();
        break;
      case "a":
        moveLeft();
        break;
      case "s":
        moveBackward();
        break;
      case "d":
        moveRight();
        break;
    }
  }

  document.addEventListener("keydown", handleKeyDown);

  return {
    update
  };
}

function createTriangleMesh() {
  const vertices = [
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
    0.0, 0.5, 0.0,
  ];
  const indices = [0, 1, 2];
  const normals = [0, 0, 1, 0, 0, 1, 0, 0, 1];
  const colors = [1, 0, 0, 1, 0, 0, 1, 0, 0];
  return {
    vertices,
    indices,
    normals,
    colors
  };
}

function optimizeRendering(gl, program, vertexBuffer, indexBuffer, indicesCount, modelMatrix, viewMatrix, projectionMatrix) {

  var mvpMatrix = mat4.create();
  mat4.multiply(mvpMatrix, projectionMatrix, viewMatrix);
  mat4.multiply(mvpMatrix, mvpMatrix, modelMatrix);
  var mvpMatrixLocation = gl.getUniformLocation(program, "u_MvpMatrix");
  gl.uniformMatrix4fv(mvpMatrixLocation, false, mvpMatrix);

  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

  var positionLocation = gl.getAttribLocation(program, "a_Position");
  var normalLocation = gl.getAttribLocation(program, "a_Normal");
  var textureLocation = gl.getAttribLocation(program, "a_TexCoord");

  gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 8 * Float32Array.BYTES_PER_ELEMENT, 0);
  gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, 8 * Float32Array.BYTES_PER_ELEMENT, 3 * Float32Array.BYTES_PER_ELEMENT);
  gl.vertexAttribPointer(textureLocation, 2, gl.FLOAT, false, 8 * Float32Array.BYTES_PER_ELEMENT, 6 * Float32Array.BYTES_PER_ELEMENT);

  gl.enableVertexAttribArray(positionLocation);
  gl.enableVertexAttribArray(normalLocation);
  gl.enableVertexAttribArray(textureLocation);

  gl.drawElements(gl.TRIANGLES, indicesCount, gl.UNSIGNED_SHORT, 0);
}

function setBackgroundColor(gl, r, g, b, a) {
  gl.clearColor(r, g, b, a);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

function createClouds(x, y, z, radius, numPoints) {
  let cloudGeometry = [];

  for (let i = 0; i < numPoints; i++) {
    let u = Math.random();
    let v = Math.random();
    let theta = 2 * Math.PI * u;
    let phi = Math.acos(2 * v - 1);

    let x = radius * Math.sin(phi) * Math.cos(theta);
    let y = radius * Math.sin(phi) * Math.sin(theta);
    let z = radius * Math.cos(phi);

    cloudGeometry.push(x);
    cloudGeometry.push(y);
    cloudGeometry.push(z);
  }

  let cloudVertices = new Float32Array(cloudGeometry);

  let cloudVertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cloudVertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloudVertices, gl.STATIC_DRAW);

  return {
    buffer: cloudVertexBuffer,
    numPoints: numPoints
  };
}

const object = {
  mass: 1,
  velocity: vec3.fromValues(0, 0, 0),
  position: vec3.fromValues(0, 0, 0)
};

function applyForce(object, force, deltaTime) {
  const acceleration = vec3.scale(vec3.create(), force, 1 / object.mass);
  vec3.add(object.velocity, object.velocity, vec3.scale(vec3.create(), acceleration, deltaTime));
  vec3.add(object.position, object.position, vec3.scale(vec3.create(), object.velocity, deltaTime));
}

function applyGravity(particles, gravity) {
  for (let i = 0; i < particles.length; i++) {
    const particle = particles[i];
    particle.velocity.y += gravity;
  }
}

function detectCollision(obj1, obj2) {

  if (obj1.maxX < obj2.minX || obj1.minX > obj2.maxX) {
    return false;
  }

  if (obj1.maxY < obj2.minY || obj1.minY > obj2.maxY) {
    return false;
  }

  if (obj1.maxZ < obj2.minZ || obj1.minZ > obj2.maxZ) {
    return false;
  }

  return true;
}

function resolveCollision(obj1, obj2) {
  const vCollision = Vector3D.subtract(obj2.position, obj1.position);
  const distance = vCollision.magnitude();

  const vCollisionNorm = Vector3D.divide(vCollision, distance);
  const vRelativeVelocity = Vector3D.subtract(obj1.velocity, obj2.velocity);

  const speed = vRelativeVelocity.dot(vCollisionNorm);

  if (speed < 0) {
    return;
  }

  const impulse = (2 * speed) / (obj1.mass + obj2.mass);

  const obj1Impulse = Vector3D.multiply(vCollisionNorm, impulse * obj2.mass);
  const obj2Impulse = Vector3D.multiply(vCollisionNorm, -impulse * obj1.mass);

  obj1.velocity = Vector3D.add(obj1.velocity, obj1Impulse);
  obj2.velocity = Vector3D.add(obj2.velocity, obj2Impulse);
}

function simulateTimeStep(objects, dt) {

  applyGravity(objects, dt);

  for (let i = 0; i < objects.length; i++) {
    for (let j = i + 1; j < objects.length; j++) {
      const obj1 = objects[i];
      const obj2 = objects[j];

      if (detectCollision(obj1, obj2)) {
        resolveCollision(obj1, obj2);
      }
    }
  }

  for (const obj of objects) {

    applyForce(obj, obj.force, dt);
    obj.force = [0, 0, 0];

    obj.position[0] += obj.velocity[0] * dt;
    obj.position[1] += obj.velocity[1] * dt;
    obj.position[2] += obj.velocity[2] * dt;
  }
}

function simulateBuoyancy(object, fluidDensity) {

  const volume = object.width * object.height * object.depth;

  const buoyantForce = volume * fluidDensity;

  const weight = object.mass * object.gravity;

  if (buoyantForce > weight) {

    object.applyForce(0, buoyantForce - weight, 0);
  }
}

function applyExternalForce(object, forceX, forceY, forceZ) {
  object.velocity.x += forceX / object.mass;
  object.velocity.y += forceY / object.mass;
  object.velocity.z += forceZ / object.mass;
}

function applyFriction(object, frictionCoefficient, deltaTime) {

  const velocityMagnitude = Math.sqrt(
    object.velocity.x * object.velocity.x +
    object.velocity.y * object.velocity.y +
    object.velocity.z * object.velocity.z
  );

  const oppositeDirection = {
    x: -object.velocity.x / velocityMagnitude,
    y: -object.velocity.y / velocityMagnitude,
    z: -object.velocity.z / velocityMagnitude
  };

  const frictionForceMagnitude = frictionCoefficient * velocityMagnitude;

  const frictionForce = {
    x: oppositeDirection.x * frictionForceMagnitude,
    y: oppositeDirection.y * frictionForceMagnitude,
    z: oppositeDirection.z * frictionForceMagnitude
  };

  object.applyForce(frictionForce.x * deltaTime, frictionForce.y * deltaTime, frictionForce.z * deltaTime);
}

function applyTorqueAndAngularAcceleration(torque, angularAcceleration, objectMatrix, inverseInertiaTensor, deltaTime) {

  const objectRotationMatrix = mat3.fromMat4(mat3.create(), objectMatrix);
  const objectAngularVelocity = vec3.transformMat3(vec3.create(), vec3.transformMat3(vec3.create(), vec3.fromValues(0, 0, 1), objectRotationMatrix), inverseInertiaTensor);

  const deltaAngularVelocity = vec3.scale(vec3.create(), torque, deltaTime).mul(inverseInertiaTensor).add(vec3.scale(vec3.create(), angularAcceleration, deltaTime));

  const newAngularVelocity = vec3.add(vec3.create(), objectAngularVelocity, deltaAngularVelocity);
  const rotationAxis = vec3.normalize(vec3.create(), newAngularVelocity);
  const rotationAngle = vec3.length(newAngularVelocity) * deltaTime;
  const rotationMatrix = mat4.fromRotation(mat4.create(), rotationAngle, rotationAxis);
  mat4.mul(objectMatrix, objectMatrix, rotationMatrix);
}

function applyRotationalForce(object, torque, duration) {

  const angularAcceleration = torque / object.momentOfInertia;

  const initialAngularVelocity = object.angularVelocity;

  const timeStep = 0.01;

  const numSteps = duration / timeStep;

  for (let i = 0; i < numSteps; i++) {

    const newAngularVelocity = initialAngularVelocity + angularAcceleration * timeStep;

    object.orientation += newAngularVelocity * timeStep;

    object.angularVelocity = newAngularVelocity;
  }
}

function applySpringForce(obj, restPos, k, damping) {

  const displacement = obj.position - restPos;

  const springForce = -k * displacement;

  const dampingForce = -damping * obj.velocity;

  const netForce = springForce + dampingForce;
  obj.applyForce(netForce);
}

function applyElectromagneticForce(obj1, obj2, k) {

  const dx = obj2.position.x - obj1.position.x;
  const dy = obj2.position.y - obj1.position.y;
  const dz = obj2.position.z - obj1.position.z;
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

  const chargeProduct = obj1.charge * obj2.charge;
  const forceMagnitude = k * chargeProduct / (distance * distance);

  const fx = forceMagnitude * dx / distance;
  const fy = forceMagnitude * dy / distance;
  const fz = forceMagnitude * dz / distance;

  obj1.applyForce(new Vector3(-fx, -fy, -fz));
  obj2.applyForce(new Vector3(fx, fy, fz));
}

function simulateGravityWells(particles, gravityWells, dt) {

  for (let i = 0; i < particles.length; i++) {
    const particle = particles[i];

    for (let j = 0; j < gravityWells.length; j++) {
      const gravityWell = gravityWells[j];

      const dx = gravityWell.x - particle.x;
      const dy = gravityWell.y - particle.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const directionX = dx / distance;
      const directionY = dy / distance;

      const forceMagnitude = gravityWell.strength * particle.mass * gravityWell.mass / distance ** 2;
      const forceX = forceMagnitude * directionX;
      const forceY = forceMagnitude * directionY;

      particle.velocity.x += forceX / particle.mass * dt;
      particle.velocity.y += forceY / particle.mass * dt;
      particle.x += particle.velocity.x * dt;
      particle.y += particle.velocity.y * dt;
    }
  }
}

function applyFracture(object, impactForce, fractureThreshold, fractureSize) {

  const magnitude = Math.sqrt(impactForce.x ** 2 + impactForce.y ** 2 + impactForce.z ** 2);

  if (magnitude > fractureThreshold) {

    const numFractures = Math.floor(magnitude / fractureThreshold);

    for (let i = 0; i < numFractures; i++) {

      const position = {
        x: object.position.x + (Math.random() * 2 - 1) * object.scale.x / 2,
        y: object.position.y + (Math.random() * 2 - 1) * object.scale.y / 2,
        z: object.position.z + (Math.random() * 2 - 1) * object.scale.z / 2,
      };

      const fracture = new THREE.Mesh(
        new THREE.BoxGeometry(fractureSize, fractureSize, fractureSize),
        object.material.clone()
      );

      fracture.position.set(position.x, position.y, position.z);
      fracture.rotation.set(Math.random() * Math.PI * 2, Math.random() * Math.PI * 2, Math.random() * Math.PI * 2);

      scene.add(fracture);

      const impulse = new THREE.Vector3(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
      impulse.normalize();
      impulse.multiplyScalar(magnitude);
      applyImpulse(fracture, impulse);
    }

    scene.remove(object);
  }
}

function applyElectricFields(objects, fields, dt) {
  for (let i = 0; i < objects.length; i++) {
    let object = objects[i];
    let q = object.charge;
    let F = [0, 0, 0];
    for (let j = 0; j < fields.length; j++) {
      let field = fields[j];
      let E = field.strength;
      let r = vec3.sub(field.position, object.position);
      let rMag = vec3.length(r);
      let rHat = vec3.normalize(r);
      let qE = vec3.scale(rHat, q * E);
      vec3.add(F, qE);
    }
    vec3.scale(F, dt);
    vec3.add(object.velocity, vec3.scale(F, object.invMass));
  }
}

function simulatePhaseTransitions(substance, temperature, pressure) {
  let phase;

  if (temperature < substance.meltingPoint) {
    phase = "solid";
  } else if (temperature > substance.boilingPoint) {
    phase = "gas";
  } else {
    phase = "liquid";
  }

  return phase;
}

function applyThermalExpansion(object, deltaTemperature, coefficientOfExpansion) {

  const expansion = object.volume * coefficientOfExpansion * deltaTemperature;

  object.width += expansion;
  object.height += expansion;
  object.depth += expansion;
}

function simulateProjectileMotion(position, velocity, gravity, timeStep) {

  position.x += velocity.x * timeStep;
  position.y += velocity.y * timeStep;
  position.z += velocity.z * timeStep;

  velocity.x += gravity.x * timeStep;
  velocity.y += gravity.y * timeStep;
  velocity.z += gravity.z * timeStep;

  return {
    position,
    velocity
  };
}

function simulateRigidBodyMotion(body, dt) {

  let force = body.forces.reduce((acc, cur) => acc.add(cur), new Vector3(0, 0, 0));
  let acceleration = force.divideScalar(body.mass);

  body.velocity.add(acceleration.multiplyScalar(dt));
  body.position.add(body.velocity.multiplyScalar(dt));

  let torque = body.torques.reduce((acc, cur) => acc.add(cur), new Vector3(0, 0, 0));
  let angularAcceleration = body.inertiaTensorInverse.multiplyVector3(torque);

  body.angularVelocity.add(angularAcceleration.multiplyScalar(dt));
  let quaternion = new Quaternion().setFromAxisAngle(body.angularVelocity.clone().normalize(), body.angularVelocity.length() * dt);
  body.orientation.multiplyQuaternions(quaternion, body.orientation);

  body.forces = [];
  body.torques = [];
}

function simulateRollingBallOnRamp(ball, ramp, deltaTime) {

  const gravity = new Vector3(0, -9.81, 0);
  const normal = ramp.getNormalAt(ball.getPosition());
  const frictionMag = ball.getFrictionCoefficient() * normal.magnitude();
  const friction = ball.getVelocity().negate().setMagnitude(frictionMag);
  const netForce = gravity.add(normal).add(friction);

  const acceleration = netForce.divide(ball.getMass());

  const newVelocity = ball.getVelocity().add(acceleration.multiply(deltaTime));
  const newPosition = ball.getPosition().add(newVelocity.multiply(deltaTime));

  const rollingDirection = ramp.getTangentAt(ball.getPosition()).cross(normal).normalize();
  const rollingVelocity = rollingDirection.multiply(ball.getRadius()).cross(newVelocity);
  const rollingAngularAcceleration = rollingVelocity.divide(ball.getRadius());
  const newAngularVelocity = ball.getAngularVelocity().add(rollingAngularAcceleration.multiply(deltaTime));
  const newRotation = ball.getRotation().add(newAngularVelocity.multiply(deltaTime));

  ball.setVelocity(newVelocity);
  ball.setPosition(newPosition);
  ball.setRotation(newRotation);
  ball.setAngularVelocity(newAngularVelocity);
}

function applyThermalConvection(fluid, temperature, gravity) {
  const cellSize = fluid.cellSize;
  const buoyancy = fluid.buoyancy;
  const ambientTemperature = fluid.ambientTemperature;

  for (let x = 0; x < fluid.width; x++) {
    for (let y = 0; y < fluid.height; y++) {
      const cell = fluid.cells[x][y];

      const deltaTemperature = temperature - cell.temperature;
      const density = cell.density;
      const buoyancyForce = density * gravity * buoyancy;

      cell.velocity.y += buoyancyForce * cellSize;
      cell.temperature += deltaTemperature * (1 - cell.smoke) * 0.1;

      if (y > 0) {
        const neighborCell = fluid.cells[x][y - 1];
        const averageTemperature = (cell.temperature + neighborCell.temperature) / 2;
        const deltaDensity = (cell.density - neighborCell.density) / density;

        cell.velocity.y += deltaDensity * averageTemperature * gravity * cellSize;
      }
    }
  }
}

function simulateBullets(bullets, gravity, airResistance, dt) {
  for (let i = 0; i < bullets.length; i++) {
    const bullet = bullets[i];

    const gravityForce = new Vector3(0, -gravity * bullet.mass, 0);
    applyForce(bullet, gravityForce);

    const velocityMagnitude = bullet.velocity.length();
    const dragMagnitude = airResistance * velocityMagnitude * velocityMagnitude;
    const dragForce = bullet.velocity.clone().multiplyScalar(-dragMagnitude);
    applyForce(bullet, dragForce);

    simulateTimeStep(bullet, dt);

    if (bullet.position.y <= 0) {
      bullet.velocity.y = -bullet.velocity.y * bullet.restitution;
    }
  }
}
