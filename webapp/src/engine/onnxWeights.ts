/**
 * Minimal ONNX protobuf parser — extracts initializer weight tensors.
 *
 * Only handles what we need: ModelProto → GraphProto → TensorProto initializers.
 * Avoids depending on a full protobuf library.
 */

export interface WeightTensor {
  name: string;
  shape: number[];
  data: Float32Array;
}

// --- Protobuf varint / wire-format helpers ---

function readVarint(buf: Uint8Array, pos: number): [value: number, newPos: number] {
  let result = 0;
  let shift = 0;
  let p = pos;
  while (p < buf.length) {
    const byte = buf[p++];
    result |= (byte & 0x7f) << shift;
    // For large varints (>28 bits), use Math to avoid sign issues
    if (shift >= 28) {
      result += (byte & 0x7f) * (2 ** shift);
      result -= (byte & 0x7f) << shift; // undo the bitwise OR
    }
    if ((byte & 0x80) === 0) break;
    shift += 7;
  }
  return [result >>> 0, p]; // unsigned
}

// Read a varint that might be 64-bit (dims can be int64)
function readVarint64(buf: Uint8Array, pos: number): [value: number, newPos: number] {
  let result = 0;
  let shift = 0;
  let p = pos;
  while (p < buf.length) {
    const byte = buf[p++];
    if (shift < 53) {
      result += (byte & 0x7f) * (2 ** shift);
    }
    if ((byte & 0x80) === 0) break;
    shift += 7;
  }
  return [result, p];
}

function readTag(buf: Uint8Array, pos: number): [fieldNumber: number, wireType: number, newPos: number] {
  const [v, p] = readVarint(buf, pos);
  return [v >>> 3, v & 7, p];
}

function readLengthDelimited(buf: Uint8Array, pos: number): [data: Uint8Array, newPos: number] {
  // Use readVarint64 (multiplication-based) to correctly handle large lengths
  // on all platforms — readVarint uses bitwise ops that overflow for lengths > 256MB
  const [len, p] = readVarint64(buf, pos);
  return [buf.subarray(p, p + len), p + len];
}

function skipField(buf: Uint8Array, pos: number, wireType: number): number {
  switch (wireType) {
    case 0: { // varint
      let p = pos;
      while (p < buf.length && (buf[p++] & 0x80) !== 0) { /* skip */ }
      return p;
    }
    case 1: return pos + 8; // 64-bit
    case 2: { // length-delimited
      const [len, p] = readVarint64(buf, pos);
      return p + len;
    }
    case 5: return pos + 4; // 32-bit
    default: throw new Error(`Unknown wire type ${wireType}`);
  }
}

// --- TensorProto parser ---

interface TensorDebug {
  name: string;
  dataType: number;
  dims: number[];
  bufLen: number;
  fieldsSeen: number[];
  rawLen: number;
  floatLen: number;
}

function parseTensorProto(buf: Uint8Array): WeightTensor | TensorDebug {
  let pos = 0;
  let name = '';
  const dims: number[] = [];
  let dataType = 0;
  let rawData: Uint8Array | null = null;
  let floatData: Float32Array | null = null;
  const fieldsSeen: number[] = [];

  while (pos < buf.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(buf, pos);
    pos = tagEnd;
    fieldsSeen.push(fieldNumber);

    switch (fieldNumber) {
      case 1: // dims (repeated int64)
        if (wireType === 2) {
          // packed
          const [packed, pEnd] = readLengthDelimited(buf, pos);
          pos = pEnd;
          let pp = 0;
          while (pp < packed.length) {
            const [dim, np] = readVarint64(packed, pp);
            dims.push(dim);
            pp = np;
          }
        } else {
          const [dim, np] = readVarint64(buf, pos);
          dims.push(dim);
          pos = np;
        }
        break;

      case 2: { // data_type
        const [dt, np] = readVarint(buf, pos);
        dataType = dt;
        pos = np;
        break;
      }

      case 4: // float_data (packed repeated float)
        if (wireType === 2) {
          const [packed, pEnd] = readLengthDelimited(buf, pos);
          pos = pEnd;
          // Ensure aligned access
          const aligned = new Uint8Array(packed.length);
          aligned.set(packed);
          floatData = new Float32Array(aligned.buffer, 0, packed.length / 4);
        } else {
          pos = skipField(buf, pos, wireType);
        }
        break;

      case 8: { // name (string)
        const [nameBytes, nEnd] = readLengthDelimited(buf, pos);
        pos = nEnd;
        name = new TextDecoder().decode(nameBytes);
        break;
      }

      case 9:   // raw_data variant used by PyTorch ONNX export (opset 17+)
      case 13: { // raw_data (bytes)
        const [raw, rEnd] = readLengthDelimited(buf, pos);
        pos = rEnd;
        rawData = raw;
        break;
      }

      default:
        pos = skipField(buf, pos, wireType);
    }
  }

  // Return debug info for non-float32 tensors or failed parses
  if (dataType !== 1 || (!rawData && !floatData)) {
    return {
      name,
      dataType,
      dims,
      bufLen: buf.length,
      fieldsSeen: fieldsSeen.slice(0, 20), // first 20 field numbers seen
      rawLen: rawData?.length ?? -1,
      floatLen: floatData?.length ?? -1,
    };
  }

  let data: Float32Array;
  if (rawData) {
    // raw_data may not be aligned — copy to aligned buffer
    const aligned = new Uint8Array(rawData.length);
    aligned.set(rawData);
    data = new Float32Array(aligned.buffer, 0, rawData.length / 4);
  } else {
    data = floatData!;
  }

  return { name, shape: dims, data };
}

function isTensor(t: WeightTensor | TensorDebug): t is WeightTensor {
  return 'shape' in t && 'data' in t;
}

// --- ModelProto / GraphProto parser ---

/** Field `field` (length-delimited) of a message, or null. Returns the first occurrence. */
function findSubmessage(buf: Uint8Array, field: number): Uint8Array | null {
  let pos = 0;
  while (pos < buf.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(buf, pos);
    pos = tagEnd;
    if (fieldNumber === field && wireType === 2) {
      return readLengthDelimited(buf, pos)[0];
    }
    pos = skipField(buf, pos, wireType);
  }
  return null;
}

/**
 * Shape of the model's first graph input (dim_value per axis; -1 for symbolic
 * dims like "batch"), read straight from the protobuf without touching weights.
 * ModelProto.graph(7) → GraphProto.input(11) → ValueInfoProto.type(2) →
 * TypeProto.tensor_type(1) → shape(2) → dim(1)* → dim_value(1) | dim_param(2).
 */
export function onnxInputShape(buffer: ArrayBuffer): number[] | null {
  const graph = findSubmessage(new Uint8Array(buffer), 7);
  if (!graph) return null;
  const input = findSubmessage(graph, 11);
  const typeProto = input && findSubmessage(input, 2);
  const tensorType = typeProto && findSubmessage(typeProto, 1);
  const shape = tensorType && findSubmessage(tensorType, 2);
  if (!shape) return null;
  const dims: number[] = [];
  let pos = 0;
  while (pos < shape.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(shape, pos);
    pos = tagEnd;
    if (fieldNumber === 1 && wireType === 2) {
      const [dim, end] = readLengthDelimited(shape, pos);
      pos = end;
      let value = -1;
      let p = 0;
      while (p < dim.length) {
        const [f, wt, te] = readTag(dim, p);
        p = te;
        if (f === 1 && wt === 0) {
          [value, p] = readVarint64(dim, p);
        } else {
          p = skipField(dim, p, wt);
        }
      }
      dims.push(value);
    } else {
      pos = skipField(shape, pos, wireType);
    }
  }
  return dims;
}

/**
 * Number of input planes the network expects: 7 (v1) or 9 (v2).
 * Reads the graph input shape [batch, C, 8, 7]; defaults to 7 if it is missing
 * or unexpected (all pre-v2 exports are 7-plane).
 */
export function onnxInputPlanes(buffer: ArrayBuffer): number {
  const shape = onnxInputShape(buffer);
  if (shape && shape.length === 4 && (shape[1] === 7 || shape[1] === 9)) return shape[1];
  return 7;
}

/**
 * Parse an ONNX model file and extract all float32 initializer tensors.
 */
export function parseOnnxWeights(buffer: ArrayBuffer): WeightTensor[] {
  const buf = new Uint8Array(buffer);
  let pos = 0;
  let graphBytes: Uint8Array | null = null;

  // Parse ModelProto — find field 7 (graph)
  while (pos < buf.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(buf, pos);
    pos = tagEnd;
    if (fieldNumber === 7 && wireType === 2) {
      const [data, end] = readLengthDelimited(buf, pos);
      graphBytes = data;
      pos = end;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }

  if (!graphBytes) throw new Error('No graph found in ONNX model');

  // Parse GraphProto — find field 5 (repeated initializer)
  const tensors: WeightTensor[] = [];
  let initCount = 0;
  let firstFail: TensorDebug | null = null;
  pos = 0;
  while (pos < graphBytes.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(graphBytes, pos);
    pos = tagEnd;
    if (fieldNumber === 5 && wireType === 2) {
      initCount++;
      const [data, end] = readLengthDelimited(graphBytes, pos);
      pos = end;
      const result = parseTensorProto(data);
      if (isTensor(result)) {
        tensors.push(result);
      } else if (!firstFail) {
        firstFail = result;
      }
    } else {
      pos = skipField(graphBytes, pos, wireType);
    }
  }

  if (tensors.length === 0) {
    throw new Error(
      `ONNX parse failed: 0 tensors from ${initCount} initializers. ` +
      `bufSize=${buf.length} graphSize=${graphBytes.length} ` +
      `firstFail=${JSON.stringify(firstFail)}`
    );
  }
  return tensors;
}
