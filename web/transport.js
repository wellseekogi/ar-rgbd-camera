const encoder = new TextEncoder();

const decoder = new TextDecoder();

export async function encodePacket(header, blob) {
  const metadata = encoder.encode(JSON.stringify(header));
  const output = new Uint8Array(4 + metadata.length + blob.size);
  new DataView(output.buffer).setUint32(0, metadata.length);
  output.set(metadata, 4);
  output.set(new Uint8Array(await blob.arrayBuffer()), 4 + metadata.length);

  return output.buffer;
}

export function decodePacket(buffer) {
  if (buffer.byteLength < 5) {
    throw new Error("서버 프레임이 너무 짧아요.");
  }

  const size = new DataView(buffer).getUint32(0);

  if (size > 4096 || 4 + size >= buffer.byteLength) {
    throw new Error("서버 프레임 헤더가 잘못됐어요.");
  }

  const header = JSON.parse(decoder.decode(new Uint8Array(buffer, 4, size)));

  return { header, blob: new Blob([buffer.slice(4 + size)], { type: "image/jpeg" }) };
}

export function socketUrl() {
  return `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
}

export async function waitForIce(peer) {
  if (peer.iceGatheringState === "complete") {
    return;
  }

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", changed);
      reject(new Error("카메라 연결 준비 시간이 초과됐어요."));
    }, 10000);

    function changed() {
      if (peer.iceGatheringState === "complete") {
        clearTimeout(timeout);
        peer.removeEventListener("icegatheringstatechange", changed);
        resolve(undefined);
      }
    }

    peer.addEventListener("icegatheringstatechange", changed);
    changed();
  });
}
