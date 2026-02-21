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
  const [len, p] = readVarint(buf, pos);
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
      const [len, p] = readVarint(buf, pos);
      return p + len;
    }
    case 5: return pos + 4; // 32-bit
    default: throw new Error(`Unknown wire type ${wireType}`);
  }
}

// --- TensorProto parser ---

function parseTensorProto(buf: Uint8Array): WeightTensor | null {
  let pos = 0;
  let name = '';
  const dims: number[] = [];
  let dataType = 0;
  let rawData: Uint8Array | null = null;
  let floatData: Float32Array | null = null;

  while (pos < buf.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(buf, pos);
    pos = tagEnd;

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

  // Only handle float32 tensors (data_type 1)
  if (dataType !== 1) return null;

  let data: Float32Array;
  if (rawData) {
    // raw_data may not be aligned — copy to aligned buffer
    const aligned = new Uint8Array(rawData.length);
    aligned.set(rawData);
    data = new Float32Array(aligned.buffer, 0, rawData.length / 4);
  } else if (floatData) {
    data = floatData;
  } else {
    return null;
  }

  return { name, shape: dims, data };
}

// --- ModelProto / GraphProto parser ---

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
  pos = 0;
  while (pos < graphBytes.length) {
    const [fieldNumber, wireType, tagEnd] = readTag(graphBytes, pos);
    pos = tagEnd;
    if (fieldNumber === 5 && wireType === 2) {
      const [data, end] = readLengthDelimited(graphBytes, pos);
      pos = end;
      const tensor = parseTensorProto(data);
      if (tensor) tensors.push(tensor);
    } else {
      pos = skipField(graphBytes, pos, wireType);
    }
  }

  return tensors;
}
