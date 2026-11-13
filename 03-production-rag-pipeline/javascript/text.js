// Python-style repr of a string, a list of strings, a number, or null, so both languages print the same text.
export function pyRepr(value) {
  if (value === null || value === undefined) return "None";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(", ")}]`;
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escaped = value.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
  return quote + (quote === "'" ? escaped.replace(/'/g, "\\'") : escaped) + quote;
}
