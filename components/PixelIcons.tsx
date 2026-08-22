interface IconProps {
  className?: string;
}

/** 8×10 blocky mic glyph, drawn as crisp-edged rects so it scales without blur. */
export function PixelMicIcon({ className = "" }: IconProps) {
  const on = "currentColor";
  // Each tuple is [x, y] on an 8-wide grid; every cell renders as a 1x1 rect.
  const cells: Array<[number, number]> = [
    [3, 0], [4, 0],
    [3, 1], [4, 1],
    [3, 2], [4, 2],
    [3, 3], [4, 3],
    [2, 4], [3, 4], [4, 4], [5, 4],
    [2, 5], [5, 5],
    [3, 6], [4, 6],
    [3, 7], [4, 7],
    [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8],
  ];
  return (
    <svg
      viewBox="0 0 8 9"
      className={className}
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      {cells.map(([x, y]) => (
        <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill={on} />
      ))}
    </svg>
  );
}

/** Blocky stop-square glyph for the "recording — tap to stop" state. */
export function PixelStopIcon({ className = "" }: IconProps) {
  return (
    <svg
      viewBox="0 0 8 8"
      className={className}
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect x={1} y={1} width={6} height={6} fill="currentColor" />
    </svg>
  );
}

/** Small pixel coin glyph used next to latency figures. */
export function PixelCoinIcon({ className = "" }: IconProps) {
  const cells: Array<[number, number]> = [
    [2, 0], [3, 0], [4, 0], [5, 0],
    [1, 1], [6, 1],
    [1, 2], [3, 2], [4, 2], [6, 2],
    [0, 3], [3, 3], [4, 3], [7, 3],
    [0, 4], [3, 4], [4, 4], [7, 4],
    [1, 5], [3, 5], [4, 5], [6, 5],
    [1, 6], [6, 6],
    [2, 7], [3, 7], [4, 7], [5, 7],
  ];
  return (
    <svg viewBox="0 0 8 8" className={className} shapeRendering="crispEdges" aria-hidden="true">
      {cells.map(([x, y]) => (
        <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill="currentColor" />
      ))}
    </svg>
  );
}

/** Pixel exclamation glyph for guardrail / warning banners. */
export function PixelWarnIcon({ className = "" }: IconProps) {
  const cells: Array<[number, number]> = [
    [3, 0], [4, 0],
    [3, 1], [4, 1],
    [3, 2], [4, 2],
    [3, 3], [4, 3],
    [3, 5], [4, 5],
    [3, 6], [4, 6],
  ];
  return (
    <svg viewBox="0 0 8 8" className={className} shapeRendering="crispEdges" aria-hidden="true">
      {cells.map(([x, y]) => (
        <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill="currentColor" />
      ))}
    </svg>
  );
}
