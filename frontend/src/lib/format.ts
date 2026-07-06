/** Small formatting helpers for the metrics UI. */

export const clamp = (v: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, v));

export const lerp = (a: number, b: number, t: number): number =>
  a + (b - a) * t;

export const fmt = (n: number, digits = 1): string =>
  n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

export const fmtInt = (n: number): string => Math.round(n).toLocaleString();

export const pct = (n: number): string => `${Math.round(n * 100)}%`;
