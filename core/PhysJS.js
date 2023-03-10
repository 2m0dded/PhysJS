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

function createLighting(gl, program, lightPosition, objectPosition, objectRotation, objectScale) {
  gl.uniform3fv(program.uniforms.lightPosition, lightPosition);
  
  var modelMatrix = mat4.create();
  mat4.translate(modelMatrix, modelMatrix, objectPosition);
  mat4.rotateX(modelMatrix, modelMatrix, objectRotation[0]);
  mat4.rotateY(modelMatrix, modelMatrix, objectRotation[1]);
  mat4.rotateZ(modelMatrix, modelMatrix, objectRotation[2]);
  mat4.scale(modelMatrix, modelMatrix, objectScale);
  gl.uniformMatrix4fv(program.uniforms.modelMatrix, false, modelMatrix);
  
  var normalMatrix = mat4.create();
  mat4.invert(normalMatrix, modelMatrix);
  mat4.transpose(normalMatrix, normalMatrix);
  gl.uniformMatrix4fv(program.uniforms.normalMatrix, false, normalMatrix);
  
  var viewMatrix = mat4.create();
  mat4.lookAt(viewMatrix, [0, 0, 5], [0, 0, 0], [0, 1, 0]);
  var projectionMatrix = mat4.create();
  mat4.perspective(projectionMatrix, 45 * Math.PI / 180, gl.canvas.width / gl.canvas.height, 0.1, 100);
  gl.uniformMatrix4fv(program.uniforms.viewMatrix, false, viewMatrix);
  gl.uniformMatrix4fv(program.uniforms.projectionMatrix, false, projectionMatrix);
  
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthFunc(gl.LEQUAL);
  gl.drawArrays(gl.TRIANGLES, 0, numVertices);
}

function simulateBouncingBall(position, velocity, gravity, groundHeight, bounceFactor, deltaTime) {
  const velocityMagnitude = vec3.length(velocity);
  const surfaceNormal = vec3.fromValues(0, 1, 0);
  
  const contactPoint = vec3.clone(position);
  contactPoint[1] = groundHeight;
  
  const ballHeight = position[1] - groundHeight;
  const isBallOnGround = ballHeight <= 0.0;
  
  if (isBallOnGround) {
    const contactNormal = vec3.clone(surfaceNormal);
    
    const relativeVelocity = vec3.clone(velocity);
    
    vec3.projectOnPlane(relativeVelocity, contactNormal, relativeVelocity);
    
    const bounceVelocityMagnitude = -relativeVelocity[1] * bounceFactor;
    
    const bounceVelocity = vec3.scale(contactNormal, contactNormal, bounceVelocityMagnitude);
    vec3.add(velocity, velocity, bounceVelocity);
    
    position[1] = groundHeight + 0.01;
  } else {
    const gravityVelocity = vec3.scale(vec3.create(), gravity, deltaTime);
    vec3.add(velocity, velocity, gravityVelocity);
  }
  
  const displacement = vec3.scale(vec3.create(), velocity, deltaTime);
  vec3.add(position, position, displacement);
  
  const dampeningFactor = 0.99;
  const dampeningVelocity = vec3.scale(vec3.create(), velocity, dampeningFactor);
  vec3.copy(velocity, dampeningVelocity);
}

function generateMipmap(gl, texture) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function drawFullscreenQuad(gl, shader, texture) {
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 1, -1, -1, 1, 1, 1,
  ]), gl.STATIC_DRAW);

  gl.useProgram(shader.program);
  gl.uniform1i(shader.uniforms.uTexture, 0);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  const positionLocation = gl.getAttribLocation(shader.program, 'aPosition');
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.disableVertexAttribArray(positionLocation);
}

async function loadOBJ(url) {
  const response = await fetch(url);
  const text = await response.text();

  const positions = [];
  const normals = [];
  const textureCoords = [];
  const indices = [];

  text.trim().split('\n').forEach(line => {
    const parts = line.trim().split(/\s+/);
    switch (parts[0]) {
      case 'v':
        positions.push(
          parseFloat(parts[1]),
          parseFloat(parts[2]),
          parseFloat(parts[3])
        );
        break;
      case 'vn':
        normals.push(
          parseFloat(parts[1]),
          parseFloat(parts[2]),
          parseFloat(parts[3])
        );
        break;
      case 'vt':
        textureCoords.push(
          parseFloat(parts[1]),
          parseFloat(parts[2])
        );
        break;
      case 'f':
        const i1 = parseInt(parts[1].split('/')[0]) - 1;
        const i2 = parseInt(parts[2].split('/')[0]) - 1;
        const i3 = parseInt(parts[3].split('/')[0]) - 1;
        indices.push(i1, i2, i3);
        break;
    }
  });

  return {
    positions,
    normals,
    textureCoords,
    indices,
  };
}

function createCubeMap(gl, images) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);

  const targets = [
    gl.TEXTURE_CUBE_MAP_POSITIVE_X,
    gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
    gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
    gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
    gl.TEXTURE_CUBE_MAP_POSITIVE_Z,
    gl.TEXTURE_CUBE_MAP_NEGATIVE_Z,
  ];

  targets.forEach((target, index) => {
    gl.texImage2D(target, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[index]);
  });

  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);

  return texture;
}

function createParticles(gl, maxParticles) {
  const vertices = new Float32Array(maxParticles * 3);
  const velocities = new Float32Array(maxParticles * 3);
  const lifeTimes = new Float32Array(maxParticles);
  const startTimes = new Float32Array(maxParticles);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  const velocityBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, velocityBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, velocities, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

  const lifeTimeBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, lifeTimeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, lifeTimes, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);

  const startTimeBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, startTimeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, startTimes, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(null);

  return {
    maxParticles,
    vertices,
    velocities,
    lifeTimes,
    startTimes,
    vao,
    update: (dt) => {
      for (let i = 0; i < maxParticles; i++) {
        if (lifeTimes[i] > 0) {
          vertices[i * 3] += velocities[i * 3] * dt;
          vertices[i * 3 + 1] += velocities[i * 3 + 1] * dt;
          vertices[i * 3 + 2] += velocities[i * 3 + 2] * dt;
          
          velocities[i * 3 + 1] -= 9.8 * dt;
          
          lifeTimes[i] -= dt;
        } else {
          vertices[i * 3] = 0;
          vertices[i * 3 + 1] = 0;
          vertices[i * 3 + 2] = 0;

          velocities[i * 3] = (Math.random() - 0.5) * 10;
          velocities[i * 3 + 1] = 5 + Math.random() * 10;
          velocities[i * 3 + 2] = (Math.random() - 0.5) * 10;

          lifeTimes[i] = Math.random() * 3;
          startTimes[i] = performance.now();
        }
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, velocityBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, velocities, gl.DYNAMIC_DRAW);
        // Update life time buffer data
      gl.bindBuffer(gl.ARRAY_BUFFER, lifeTimeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, lifeTimes, gl.DYNAMIC_DRAW);

  // Update start time buffer data
      gl.bindBuffer(gl.ARRAY_BUFFER, startTimeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, startTimes, gl.DYNAMIC_DRAW);
    },
    draw: (program) => {
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.POINTS, 0, maxParticles);
      gl.bindVertexArray(null);
    },
  };
}

function createBaseplate(gl, size, color) {
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  const positions = [
    size, 0, -size,
    -size, 0, -size,
    size, 0, size,
    -size, 0, size,
  ];

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);

  const colors = [
    ...color, ...color, ...color, ...color,
  ];

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  gl.enableVertexAttribArray(0);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  gl.enableVertexAttribArray(1);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(null);

  return {
    draw: (program) => {
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
    },
  };
}

function createUIElement(gl, position, size, color, texture) {
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  const positions = [
    position[0] + size[0], position[1] + size[1], 0.0,
    position[0], position[1] + size[1], 0.0,
    position[0] + size[0], position[1], 0.0,
    position[0], position[1], 0.0,
  ];

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  const texCoordBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);

  const texCoords = [
    1.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    0.0, 0.0,
  ];

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);

  const colorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);

  const colors = [
    ...color, ...color, ...color, ...color,
  ];

  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  gl.enableVertexAttribArray(0);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  gl.enableVertexAttribArray(1);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

  gl.enableVertexAttribArray(2);
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(null);

  return {
    position,
    size,
    texture,
    draw: (program) => {
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.bindVertexArray(null);
    },
    contains: (point) => {
      const minX = position[0];
      const maxX = position[0] + size[0];
      const minY = position[1];
      const maxY = position[1] + size[1];
      return point[0] >= minX && point[0] <= maxX && point[1] >= minY && point[1] <= maxY;
    },
  };
}

function cleanup(gl, program, buffers, textures, framebuffers) {
  gl.useProgram(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindVertexArray(null);

  if (buffers) {
    buffers.forEach((buffer) => {
      gl.deleteBuffer(buffer);
    });
  }

  if (textures) {
    textures.forEach((texture) => {
      gl.deleteTexture(texture);
    });
  }

  if (framebuffers) {
    framebuffers.forEach((framebuffer) => {
      gl.deleteFramebuffer(framebuffer);
    });
  }

  if (program) {
    gl.deleteProgram(program);
  }

  gl.clearColor(0.0, 0.0, 0.0, 0.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
}

function closeGame(gl, program, buffers, textures, framebuffers) {
  cleanup(gl, program, buffers, textures, framebuffers);
}

function connectToSQLServer(config) {
  const socket = new WebSocket(`ws://${config.host}:${config.port}`);

  socket.onopen = () => {
    console.log('Connected to SQL server!');
    socket.send(`USE ${config.database};`);
  };

  socket.onerror = (err) => {
    console.error('Error connecting to SQL server:', err);
    socket.close();
  };

  socket.onclose = () => {
    console.log('Disconnected from SQL server.');
    socket.close();
  };

  return socket;
}

function cacheGameContent(key, content) {
  try {
    const serializedContent = JSON.stringify(content);
    localStorage.setItem(key, serializedContent);
    console.log(`Cached content for key "${key}".`);
  } catch (err) {
    console.error(`Error caching content for key "${key}":`, err);
  }
}

function getCachedGameContent(key) {
  try {
    const serializedContent = localStorage.getItem(key);
    const content = JSON.parse(serializedContent);
    console.log(`Retrieved cached content for key "${key}".`);
    return content;
  } catch (err) {
    console.error(`Error retrieving cached content for key "${key}":`, err);
    return null;
  }
}

function simulateGlassBreak(numParticles, center, radius, velocity) {
  const particles = [];

  for (let i = 0; i < numParticles; i++) {
    const position = [
      center[0] + (Math.random() - 0.5) * radius,
      center[1] + (Math.random() - 0.5) * radius,
      center[2] + (Math.random() - 0.5) * radius,
    ];
    const mass = 0.1;
    const acceleration = [0, -9.81, 0];
    const particle = { position, velocity, acceleration, mass };
    particles.push(particle);
  }

  const timeStep = 0.01;
  const numSteps = 1000;
  for (let i = 0; i < numSteps; i++) {
    for (let j = 0; j < numParticles; j++) {
      const particle = particles[j];
      particle.position[0] += particle.velocity[0] * timeStep;
      particle.position[1] += particle.velocity[1] * timeStep;
      particle.position[2] += particle.velocity[2] * timeStep;
    }

    for (let j = 0; j < numParticles; j++) {
      const particle = particles[j];
      particle.velocity[0] += particle.acceleration[0] * timeStep;
      particle.velocity[1] += particle.acceleration[1] * timeStep;
      particle.velocity[2] += particle.acceleration[2] * timeStep;
    }
  }

  return particles.map(particle => particle.position);
}

function loadAudio(src) {
  const audio = new Audio();
  audio.src = src;
  return audio;
}

function playAudio(audio) {
  audio.play();
}


function pauseAudio(audio) {
  audio.pause();
}

function stopAudio(audio) {
  audio.pause();
  audio.currentTime = 0;
}

function setAudioVolume(audio, volume) {
  audio.volume = volume;
}

function createProximityAudio(src, maxDistance, fadeInTime, fadeOutTime) {
  const audio = new Audio(src);
  audio.loop = true;
  audio.volume = 0;

  const audioContext = new AudioContext();
  const source = audioContext.createMediaElementSource(audio);
  const gainNode = audioContext.createGain();
  source.connect(gainNode);
  gainNode.connect(audioContext.destination);

  function updateVolume(position) {
    const distance = Math.sqrt(
      Math.pow(position.x - audio.position.x, 2) +
      Math.pow(position.y - audio.position.y, 2) +
      Math.pow(position.z - audio.position.z, 2)
    );
    let volume = 1 - (distance / maxDistance);
    volume = Math.max(0, Math.min(1, volume));
    gainNode.gain.setValueAtTime(volume, audioContext.currentTime);

    if (volume > audio.volume) {
      audio.volume += (volume - audio.volume) / fadeInTime;
    } else {
      audio.volume -= (audio.volume - volume) / fadeOutTime;
    }
  }

  return { audio, updateVolume };
}

function vectorNorm(u) {
  return Math.sqrt(dotProduct(u, u));
}

function fft(signal) {
  const n = signal.length;
  if (n === 1) {
    return signal;
  }
  const even = new Array(n / 2);
  const odd = new Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    even[i] = signal[2 * i];
    odd[i] = signal[2 * i + 1];
  }
  const spectrumEven = fft(even);
  const spectrumOdd = fft(odd);
  const spectrum = new Array(n);
  for (let i = 0; i < n / 2; i++) {
    const angle = -2 * Math.PI * i / n;
    const twiddleReal = Math.cos(angle);
    const twiddleImag = Math.sin(angle);
    const re = spectrumEven[i][0];
    const im = spectrumEven[i][1];
    const productRe = re * twiddleReal - im * twiddleImag;
    const productIm = re * twiddleImag + im * twiddleReal;
    spectrum[i] = [productRe + spectrumOdd[i][0], productIm + spectrumOdd[i][1]];
    spectrum[i + n / 2] = [productRe - spectrumOdd[i][0], productIm - spectrumOdd[i][1]];
  }
  return spectrum;
}

function calculateWaveFunction(x, t, V, m, hbar) {
  
  const psi0 = /* initial wave function */;
  
  const dx = /* step size */;
  const dt = /* time step size */;
  
  const psi = [psi0];
  
  for (let n = 1; n <= t/dt; n++) {
    const psi_n = psi[n-1];
    
    // Iterate over positions
    for (let i = 1; i < x.length-1; i++) {
      const x_i = x[i];
      const V_i = V(x_i);
      
      const psi_i = psi_n[i];
      const psi_iplus1 = psi_n[i+1];
      const psi_iminus1 = psi_n[i-1];
      
      const k = 2*m/hbar**2 * (V_i - E);
      
      const psi_i_new = psi_i + (k*dt/2) * (psi_iplus1 - 2*psi_i + psi_iminus1);
      psi[n][i] = psi_i_new;
    }
  }
  
  return psi;
}

function calculateSpinWaveDispersion(kx, ky, J, S, M) {
  const N = kx.length;
  const omega = [];
  
  for (let i = 0; i < N; i++) {
    const k = Math.sqrt(kx[i]**2 + ky[i]**2);
    
    const J_k = J(k);
    const w = S * Math.sqrt(J_k * (J_k + M));
    omega.push(w);
  }
  
  return omega;
}

function calculateLambShift(Z, n, l, j, S) {
  const alpha = 1/137;
  const Ry = 13.605693122994;
  const a0 = 0.52917721067e-10;
  const En = - Ry / (n**2);
  const E1 = - Ry / 4;
  const DeltaE = alpha**2 * Ry / (2*n**3) * (
    1/(j+0.5) - 3/4*n/(j+0.5)**2
    + l*(l+1)-j*(j+1)-S*(S+1)/(j+0.5)/(j+1.5)
  );
  
  return DeltaE;
}

function setBit(n, i) {
  return n | (1 << i);
}

function clearBit(n, i) {
  return n & ~(1 << i);
}

function toggleBit(n, i) {
  return n ^ (1 << i);
}

function getBit(n, i) {
  return (n >> i) & 1;
}

function countBits(n) {
  let count = 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

function rotateLeft(n, k) {
  return ((n << k) | (n >>> (32 - k))) >>> 0;
}

function rotateRight(n, k) {
  return ((n >>> k) | (n << (32 - k))) >>> 0;
}

function calculateDiffuseLight(color, normal, lightDirection) {
  const dotProduct = Math.max(0, normal.dot(lightDirection));
  return color.multiplyScalar(dotProduct);
}

function calculateSpecularLight(color, normal, lightDirection, viewDirection, shininess) {
  const reflectionDirection = lightDirection.clone().reflect(normal);
  const dotProduct = Math.max(0, reflectionDirection.dot(viewDirection));
  return color.multiplyScalar(Math.pow(dotProduct, shininess));
}

function calculateAmbientLight(color, ambientColor) {
  return color.multiply(ambientColor);
}

function calculateFresnelReflection(cosTheta, eta1, eta2) {
  const sinThetaT = (eta1 / eta2) * Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  if (sinThetaT >= 1) {
    return 1;
  } else {
    const cosThetaT = Math.sqrt(Math.max(0, 1 - sinThetaT * sinThetaT));
    const rPerp = ((eta2 * cosTheta - eta1 * cosThetaT) / (eta2 * cosTheta + eta1 * cosThetaT)) ** 2;
    const rParallel = ((eta1 * cosTheta - eta2 * cosThetaT) / (eta1 * cosTheta + eta2 * cosThetaT)) ** 2;
    return 0.5 * (rPerp + rParallel);
  }
}

function calculateBlinnPhongBRDF(lightDirection, viewDirection, normal, specularColor, shininess) {
  const halfwayDirection = lightDirection.clone().add(viewDirection).normalize();
  const cosTheta = normal.dot(halfwayDirection);
  const specular = specularColor.multiplyScalar(Math.pow(Math.max(0, cosTheta), shininess));
  const diffuse = new THREE.Color(0, 0, 0);
  return { diffuse, specular };
}

function calculateSchlickApproximation(cosTheta, refractionIndex) {
  const r0 = ((1 - refractionIndex) / (1 + refractionIndex)) ** 2;
  return r0 + (1 - r0) * Math.pow(1 - cosTheta, 5);
}

function calculateBeerLambertLaw(color, distance, absorptionCoefficient) {
  return color.clone().multiplyScalar(Math.exp(-absorptionCoefficient * distance));
}

function calculateSnellReflection(direction, normal, refractionIndex1, refractionIndex2) {
  const cosTheta1 = -normal.dot(direction);
  const sinTheta1 = Math.sqrt(Math.max(0, 1 - cosTheta1 * cosTheta1));
  const sinTheta2 = refractionIndex1 / refractionIndex2 * sinTheta1;
  if (sinTheta2 >= 1) {
    // Total internal reflection
    return direction.clone().reflect(normal);
  } else {
    const cosTheta2 = Math.sqrt(Math.max(0, 1 - sinTheta2 * sinTheta2));
    return direction.clone().multiplyScalar(refractionIndex1 / refractionIndex2).add(normal.clone().multiplyScalar(refractionIndex1 / refractionIndex2 * cosTheta1 - cosTheta2)).normalize();
  }
}

function lerp(start, end, time) {
  return (1 - time) * start + time * end;
}

function ease(start, end, time, easingFn = (t) => t) {
  return (end - start) * easingFn(time) + start;
}

function animate(updateFn, fps) {
  const frameInterval = 1000 / fps; // Calculate time between each frame
  let lastFrameTime = 0;

  function loop(currentTime) {
    const deltaTime = currentTime - lastFrameTime;

    if (deltaTime >= frameInterval) {
      updateFn(deltaTime);
      lastFrameTime = currentTime;
    }

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

function rotatePoint(point, origin, angle) {
  const s = Math.sin(angle);
  const c = Math.cos(angle);

  const translatedX = point.x - origin.x;
  const translatedY = point.y - origin.y;

  const rotatedX = translatedX * c - translatedY * s;
  const rotatedY = translatedX * s + translatedY * c;

  return {
    x: rotatedX + origin.x,
    y: rotatedY + origin.y,
  };
}

function lerpVec2(start, end, time) {
  const x = lerp(start.x, end.x, time);
  const y = lerp(start.y, end.y, time);

  return { x, y };
}

function easeVec2(start, end, time, easingFn = (t) => t) {
  const x = ease(start.x, end.x, time, easingFn);
  const y = ease(start.y, end.y, time, easingFn);

  return { x, y };
}

function rotateVec2(vec, origin, angle) {
  const { x, y } = vec;
  const s = Math.sin(angle);
  const c = Math.cos(angle);

  const translatedX = x - origin.x;
  const translatedY = y - origin.y;

  const rotatedX = translatedX * c - translatedY * s;
  const rotatedY = translatedX * s + translatedY * c;

  return {
    x: rotatedX + origin.x,
    y: rotatedY + origin.y,
  };
}

function lerpColor(start, end, time) {
  const r = lerp(start.r, end.r, time);
  const g = lerp(start.g, end.g, time);
  const b = lerp(start.b, end.b, time);

  return { r, g, b };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function distanceVec2(a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;

  return Math.sqrt(dx * dx + dy * dy);
}

function dotVec2(a, b) {
  return a.x * b.x + a.y * b.y;
}

function normalizeVec2(vec) {
  const length = Math.sqrt(vec.x * vec.x + vec.y * vec.y);

  if (length === 0) {
    return { x: 0, y: 0 };
  }

  return { x: vec.x / length, y: vec.y / length };
}

function lerpAngle(start, end, time) {
  let difference = end - start;

  if (difference > Math.PI) {
    difference -= 2 * Math.PI;
  } else if (difference < -Math.PI) {
    difference += 2 * Math.PI;
  }

  return start + difference * time;
}

function deepCopy(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function isObjectEmpty(obj) {
  return Object.keys(obj).length === 0;
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function isNumberInRange(num, min, max) {
  return num >= min && num <= max;
}

function getRandomElement(array) {
  return array[Math.floor(Math.random() * array.length)];
}

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function easeInOutQuad(t, b, c, d) {
  t /= d / 2;
  if (t < 1) {
    return c / 2 * t * t + b;
  }
  t--;
  return -c / 2 * (t * (t - 2) - 1) + b;
}

function sortObjectsByKey(objects, key) {
  objects.sort(function(a, b) {
    const valueA = a[key];
    const valueB = b[key];
    if (valueA < valueB) {
      return -1;
    }
    if (valueA > valueB) {
      return 1;
    }
    return 0;
  });
}

function isPowerOfTwo(num) {
  return (num & (num - 1)) === 0;
}

function roundToNearest(num, nearest) {
  return Math.round(num / nearest) * nearest;
}

function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

function setCookie(name, value, days) {
  const date = new Date();
  date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
  const expires = "expires=" + date.toUTCString();
  document.cookie = name + "=" + value + ";" + expires + ";path=/";
}

function getCookie(name) {
  const cookieName = name + "=";
  const cookies = document.cookie.split(";");
  for (let i = 0; i < cookies.length; i++) {
    let cookie = cookies[i];
    while (cookie.charAt(0) == " ") {
      cookie = cookie.substring(1);
    }
    if (cookie.indexOf(cookieName) == 0) {
      return cookie.substring(cookieName.length, cookie.length);
    }
  }
  return "";
}

function setLocalStorageItem(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function getLocalStorageItem(key) {
  const value = localStorage.getItem(key);
  return value ? JSON.parse(value) : null;
}

function setSessionStorageItem(key, value) {
  sessionStorage.setItem(key, JSON.stringify(value));
}

function getSessionStorageItem(key) {
  const value = sessionStorage.getItem(key);
  return value ? JSON.parse(value) : null;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function getScreenWidth() {
  return window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
}

function getScreenHeight() {
  return window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
}

function simulateFlashbang(position, intensity, duration, objects) {
  const circle = new Circle(position, intensity);  
  for (let i = 0; i < objects.length; i++) {
    const object = objects[i];
    if (circle.contains(object.position)) {
      object.isBlinded = true;
      object.blindDuration = duration;
    }
  }
}

function simulateExplosion(position, radius, objects) {
  for (let i = 0; i < objects.length; i++) {
    const object = objects[i];
    const distance = position.distanceTo(object.position);
    if (distance <= radius) {
      const force = (1 - distance / radius) * 10;
      const direction = object.position.clone().sub(position).normalize();
      object.velocity.add(direction.multiplyScalar(force));
      object.isHit = true;
    }
  }
}

class Weapon {
  constructor(name, ammoType, magazineSize, maxAmmo) {
    this.name = name;
    this.ammoType = ammoType;
    this.magazineSize = magazineSize;
    this.maxAmmo = maxAmmo;
    this.magazine = magazineSize;
    this.ammo = maxAmmo - magazineSize;
    this.fireMode = "single";
    this.fireRate = 500;
    this.lastFired = 0;
  }
  
  reload() {
    const missingAmmo = this.magazineSize - this.magazine;
    if (this.ammo >= missingAmmo) {
      this.magazine += missingAmmo;
      this.ammo -= missingAmmo;
    } else {
      this.magazine += this.ammo;
      this.ammo = 0;
    }
  }
  
  fire() {
    if (this.magazine > 0) {
      if (Date.now() - this.lastFired >= this.fireRate) {
        switch (this.fireMode) {
          case "single":
            this.fireSingle();
            break;
          case "burst":
            this.fireBurst();
            break;
          case "auto":
            this.fireAuto();
            break;
        }
        this.lastFired = Date.now();
        this.magazine--;
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  
  fireSingle() {
    // Perform the firing action for a single shot here...
  }
  
  fireBurst() {
    // Perform the firing action for a burst of shots here...
  }
  
  fireAuto() {
    // Perform the firing action for full-auto fire here...
  }
  
  switchFireMode() {
    switch (this.fireMode) {
      case "single":
        this.fireMode = "burst";
        break;
      case "burst":
        this.fireMode = "auto";
        break;
      case "auto":
        this.fireMode = "single";
        break;
    }
    console.log(`Switched to ${this.fireMode} mode!`);
  }
  
  addAmmo(amount) {
    const totalAmmo = this.ammo + amount;
    if (totalAmmo > this.maxAmmo) {
      this.ammo = this.maxAmmo;
    } else {
      this.ammo = totalAmmo;
    }
  }
}

function createTracer(startPos, endPos, color, width) {
  let vertices = [
    startPos.x, startPos.y, startPos.z,
    endPos.x, endPos.y, endPos.z
  ];
  let vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  let shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
  gl.useProgram(shaderProgram);
  gl.uniformMatrix4fv(shaderProgram.uModelViewMatrix, false, modelViewMatrix);
  gl.uniformMatrix4fv(shaderProgram.uProjectionMatrix, false, projectionMatrix);
  gl.uniform4f(shaderProgram.uColor, color.r, color.g, color.b, color.a);
  gl.uniform1f(shaderProgram.uWidth, width);
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.vertexAttribPointer(shaderProgram.aVertexPosition, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(shaderProgram.aVertexPosition);
  gl.drawArrays(gl.LINES, 0, vertices.length / 3);
}

function reconstructPath(endNode) {
  let path = [endNode];
  let currentNode = endNode;
  while (currentNode.parent) {
    path.push(currentNode.parent);
    currentNode = currentNode.parent;
  }
  return path.reverse();
}

function distance(node1, node2) {
  let dx = node1.x - node2.x;
  let dy = node1.y - node2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function heuristic(node1, node2) {
  return Math.abs(node1.x - node2.x) + Math.abs(node1.y - node2.y);
}

function findPath(startNode, endNode, graph) {
  let openList = [startNode];
  let closedList = [];
  startNode.gScore = 0;
  startNode.fScore = heuristic(startNode, endNode);
  while (openList.length > 0) {
    let current = openList.reduce((minNode, node) => node.fScore < minNode.fScore ? node : minNode);
    if (current === endNode) {
      return reconstructPath(endNode);
    }
    openList = openList.filter(node => node !== current);
    closedList.push(current);
    for (let neighbor of current.neighbors) {
      if (closedList.includes(neighbor)) {
        continue;
      }      
      let tentativeGScore = current.gScore + distance(current, neighbor);      
      if (!openList.includes(neighbor)) {
        openList.push(neighbor);
      } else if (tentativeGScore >= neighbor.gScore) {
        continue;
      }
      neighbor.parent = current;
      neighbor.gScore = tentativeGScore;
      neighbor.fScore = tentativeGScore + heuristic(neighbor, endNode);
    }
  }  
  return null;
}

function attachAttachment2D(attachment, weapon) {
  weapon.attachments.push(attachment);
  if (attachment.type === "scope") {
    weapon.accuracy += 10;
    weapon.range += 50;
  } else if (attachment.type === "silencer") {
    weapon.noise -= 5;
  } else if (attachment.type === "extended_magazine") {
    weapon.magazine_size *= 2;
  }
  weapon.sprite.src = "weapon_with_" + attachment.type + ".png";
}

function mapRange(value, inMin, inMax, outMin, outMax) {
  return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}
