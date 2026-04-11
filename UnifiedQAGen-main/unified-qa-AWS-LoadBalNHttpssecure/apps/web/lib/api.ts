const browserBase =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "";

const serverBase =
  process.env.INTERNAL_API_BASE_URL || "http://api:8000";

function getApiBase() {
  return typeof window === "undefined" ? serverBase : browserBase;
}
export async function createJob(payload: {
  url: string;
  strictness: string;
  auto_mode: boolean;
  requested_pairs: number;
}) {
  const res = await fetch(`${getApiBase()}/api/v1/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Create job failed: ${res.status}`);
  }

  return res.json();
}

export async function getJob(jobId: string) {
  const res = await fetch(`${getApiBase()}/api/v1/jobs/${jobId}`, {
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Get job failed: ${res.status}`);
  }

  return res.json();
}

export async function getJobResults(jobId: string) {
  const res = await fetch(`${getApiBase()}/api/v1/jobs/${jobId}/results`, {
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Get job results failed: ${res.status}`);
  }

  return res.json();
}

export function getJobResultsJsonUrl(jobId: string) {
  return `${browserBase}/api/v1/jobs/${jobId}/results.json`;
}

export function getJobResultsCsvUrl(jobId: string) {
  return `${browserBase}/api/v1/jobs/${jobId}/results.csv`;
}
