"use client";

import { useMemo, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "";
type Pair = {
  question: string;
  answer: string;
  context?: string;
  source?: string;
  type?: string;
  quality_score?: number;
};

type EvaluationResponse = {
  title?: string;
  status?: string;
  method?: string;
  training_grade_pairs: number;
  raw_extractable_pairs: number;
  predicted_min?: number;
  predicted_max?: number;
  confidence?: number;
  quality_view?: unknown;
  metrics?: unknown;
};

type CreateJobResponse = {
  job_id: string;
  status: string;
};

type JobStatusResponse = {
  job_id: string;
  status: string;
  progress: number;
  step: string;
  summary?: string | null;
  error_message?: string | null;
};

type JobResultsResponse = {
  job_id: string;
  status: string;
  summary?: string | null;
  capacity?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  accepted_pairs?: Pair[] | null;
  artifacts?: Record<string, unknown> | null;
};

const STRICTNESS_OPTIONS = [
  { label: "Strict", evaluatorValue: "Strict", jobValue: "strict" },
  { label: "Standard", evaluatorValue: "Standard", jobValue: "medium" },
  { label: "Lenient", evaluatorValue: "Lenient", jobValue: "lenient" },
] as const;

type StrictnessLabel = (typeof STRICTNESS_OPTIONS)[number]["label"];

export default function EvaluatorPage() {
  const [url, setUrl] = useState("");
  const [strictness, setStrictness] = useState<StrictnessLabel>("Standard");

  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [generating, setGenerating] = useState(false);

  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [jobResult, setJobResult] = useState<JobResultsResponse | null>(null);

  const [jobId, setJobId] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  const [jobStep, setJobStep] = useState("");

  const [error, setError] = useState("");

  const strictnessConfig =
    STRICTNESS_OPTIONS.find((option) => option.label === strictness) ??
    STRICTNESS_OPTIONS[1];

  const generatedPairs: Pair[] = useMemo(() => {
    if (!jobResult?.accepted_pairs) return [];
    return jobResult.accepted_pairs;
  }, [jobResult]);

  const sleep = (ms: number) =>
    new Promise((resolve) => setTimeout(resolve, ms));

  const getErrorMessage = (value: unknown, fallback: string) => {
    if (typeof value === "string") return value;
    if (value && typeof value === "object") return JSON.stringify(value);
    return fallback;
  };

  const onEvaluate = async (e: React.FormEvent) => {
    e.preventDefault();

    setEvaluating(true);
    setLoading(true);
    setError("");

    setResult(null);
    setJobResult(null);
    setJobStatus("");
    setJobProgress(0);
    setJobStep("");
    setJobId("");

    try {
      const res = await fetch(`${API_BASE}/api/v1/evaluator`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          strictness: strictnessConfig.evaluatorValue,
        }),
      });

      const data: EvaluationResponse | { detail?: unknown } = await res.json();

      if (!res.ok) {
        throw new Error(
          "detail" in data
            ? getErrorMessage(data.detail, "Evaluation failed")
            : "Evaluation failed"
        );
      }

      setResult(data as EvaluationResponse);
    } catch (err: any) {
      setError(getErrorMessage(err?.message ?? err, "Something went wrong"));
    } finally {
      setEvaluating(false);
      setLoading(false);
    }
  };

  const fetchJobStatus = async (createdJobId: string): Promise<JobStatusResponse> => {
    const res = await fetch(`${API_BASE}/api/v1/jobs/${createdJobId}`, {
      cache: "no-store",
    });

    const data: JobStatusResponse | { detail?: unknown } = await res.json();

    if (!res.ok) {
      throw new Error(
        "detail" in data
          ? getErrorMessage(data.detail, "Failed to fetch job status")
          : "Failed to fetch job status"
      );
    }

    return data as JobStatusResponse;
  };

  const fetchJobResults = async (
    createdJobId: string
  ): Promise<JobResultsResponse> => {
    const res = await fetch(`${API_BASE}/api/v1/jobs/${createdJobId}/results`, {
      cache: "no-store",
    });

    const data: JobResultsResponse | { detail?: unknown } = await res.json();

    if (!res.ok) {
      throw new Error(
        "detail" in data
          ? getErrorMessage(data.detail, "Failed to fetch job results")
          : "Failed to fetch job results"
      );
    }

    return data as JobResultsResponse;
  };

  const pollJobUntilComplete = async (createdJobId: string) => {
    const maxAttempts = 180;
    const delayMs = 2000;

    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const job = await fetchJobStatus(createdJobId);

      setJobStatus(job.status || "");
      setJobProgress(job.progress ?? 0);
      setJobStep(job.step || "");

      if (job.status === "completed") {
        const results = await fetchJobResults(createdJobId);
        setJobResult(results);
        setJobStatus(results.status || "completed");
        setError("");
        return;
      }

      if (job.status === "failed" || job.status === "error") {
        throw new Error(job.error_message || "Job failed");
      }

      await sleep(delayMs);
    }

    throw new Error(
      "Job is still running. Please open the job page or refresh in a moment."
    );
  };

  const onCreateJob = async () => {
    if (!result) return;

    setGenerating(true);
    setLoading(true);
    setError("");

    setJobResult(null);
    setJobStatus("");
    setJobProgress(0);
    setJobStep("");
    setJobId("");

    try {
      const res = await fetch(`${API_BASE}/api/v1/jobs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          strictness: strictnessConfig.jobValue,
          auto_mode: true,
          requested_pairs: result.training_grade_pairs,
          evaluation: {
            training_grade_pairs: result.training_grade_pairs,
            raw_extractable_pairs: result.raw_extractable_pairs,
          },
        }),
      });

      const data: CreateJobResponse | { detail?: unknown } = await res.json();

      if (!res.ok) {
        throw new Error(
          "detail" in data
            ? getErrorMessage(data.detail, "Job creation failed")
            : "Job creation failed"
        );
      }

      if (!("job_id" in data) || !data.job_id) {
        throw new Error("Job ID missing from create job response");
      }

      setJobId(data.job_id);
      setJobStatus(data.status || "queued");

      await pollJobUntilComplete(data.job_id);
    } catch (err: any) {
      setError(getErrorMessage(err?.message ?? err, "Job creation failed"));
    } finally {
      setGenerating(false);
      setLoading(false);
    }
  };

  const downloadJsonUrl = jobId
    ? `${API_BASE}/api/v1/jobs/${jobId}/results.json`
    : "";

  const downloadCsvUrl = jobId
    ? `${API_BASE}/api/v1/jobs/${jobId}/results.csv`
    : "";

  const jobPageUrl = jobId ? `/jobs/${jobId}` : "";

  const styles = {
    page: {
      minHeight: "100vh",
      background: "linear-gradient(180deg, #0b1020 0%, #11162a 45%, #0f172a 100%)",
      color: "#e5e7eb",
      padding: "32px 20px 64px",
      fontFamily:
        'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    } as React.CSSProperties,

    container: {
      maxWidth: 1180,
      margin: "0 auto",
    } as React.CSSProperties,

    hero: {
      marginBottom: 24,
      padding: 24,
      borderRadius: 20,
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      boxShadow: "0 10px 40px rgba(0,0,0,0.25)",
      backdropFilter: "blur(8px)",
    } as React.CSSProperties,

    title: {
      margin: 0,
      fontSize: 34,
      fontWeight: 800,
      letterSpacing: "-0.03em",
    } as React.CSSProperties,

    subtitle: {
      marginTop: 10,
      marginBottom: 0,
      color: "#9ca3af",
      fontSize: 15,
      lineHeight: 1.6,
    } as React.CSSProperties,

    formCard: {
      display: "grid",
      gap: 14,
      padding: 20,
      borderRadius: 20,
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      marginBottom: 24,
      boxShadow: "0 10px 30px rgba(0,0,0,0.2)",
    } as React.CSSProperties,

    label: {
      fontSize: 13,
      fontWeight: 600,
      color: "#cbd5e1",
      marginBottom: 6,
      display: "block",
    } as React.CSSProperties,

    input: {
      width: "100%",
      padding: "14px 16px",
      borderRadius: 14,
      border: "1px solid rgba(255,255,255,0.1)",
      background: "rgba(15,23,42,0.8)",
      color: "#f8fafc",
      outline: "none",
      fontSize: 15,
    } as React.CSSProperties,

    select: {
      width: 220,
      padding: "14px 16px",
      borderRadius: 14,
      border: "1px solid rgba(255,255,255,0.1)",
      background: "rgba(15,23,42,0.8)",
      color: "#f8fafc",
      outline: "none",
      fontSize: 15,
    } as React.CSSProperties,

    buttonRow: {
      display: "flex",
      gap: 12,
      flexWrap: "wrap",
      marginTop: 4,
    } as React.CSSProperties,

    primaryButton: {
      padding: "13px 18px",
      borderRadius: 14,
      border: "none",
      background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
      color: "white",
      fontWeight: 700,
      cursor: "pointer",
      minWidth: 180,
      boxShadow: "0 8px 24px rgba(59,130,246,0.35)",
    } as React.CSSProperties,

    secondaryButton: {
      padding: "13px 18px",
      borderRadius: 14,
      border: "1px solid rgba(255,255,255,0.12)",
      background: "rgba(255,255,255,0.06)",
      color: "white",
      fontWeight: 700,
      cursor: "pointer",
      minWidth: 180,
    } as React.CSSProperties,

    linkButton: {
      display: "inline-block",
      padding: "13px 18px",
      borderRadius: 14,
      border: "1px solid rgba(255,255,255,0.12)",
      background: "rgba(255,255,255,0.06)",
      color: "white",
      fontWeight: 700,
      textDecoration: "none",
      minWidth: 180,
      textAlign: "center" as const,
    } as React.CSSProperties,

    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
      gap: 16,
      marginBottom: 24,
    } as React.CSSProperties,

    metricCard: {
      padding: 18,
      borderRadius: 18,
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      boxShadow: "0 8px 24px rgba(0,0,0,0.16)",
    } as React.CSSProperties,

    metricLabel: {
      color: "#94a3b8",
      fontSize: 13,
      marginBottom: 8,
    } as React.CSSProperties,

    metricValue: {
      fontSize: 28,
      fontWeight: 800,
      color: "#f8fafc",
      letterSpacing: "-0.03em",
    } as React.CSSProperties,

    section: {
      marginTop: 20,
      padding: 20,
      borderRadius: 20,
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      boxShadow: "0 10px 30px rgba(0,0,0,0.2)",
    } as React.CSSProperties,

    sectionTitle: {
      marginTop: 0,
      marginBottom: 16,
      fontSize: 20,
      fontWeight: 800,
      letterSpacing: "-0.02em",
    } as React.CSSProperties,

    infoRow: {
      display: "grid",
      gridTemplateColumns: "180px 1fr",
      gap: 10,
      padding: "10px 0",
      borderBottom: "1px solid rgba(255,255,255,0.06)",
    } as React.CSSProperties,

    infoLabel: {
      color: "#94a3b8",
      fontWeight: 600,
      fontSize: 14,
    } as React.CSSProperties,

    infoValue: {
      color: "#f8fafc",
      fontSize: 14,
      wordBreak: "break-word" as const,
    } as React.CSSProperties,

    success: {
      padding: "14px 16px",
      borderRadius: 14,
      background: "rgba(34,197,94,0.12)",
      border: "1px solid rgba(34,197,94,0.3)",
      color: "#bbf7d0",
      marginBottom: 16,
    } as React.CSSProperties,

    warning: {
      padding: "14px 16px",
      borderRadius: 14,
      background: "rgba(245,158,11,0.12)",
      border: "1px solid rgba(245,158,11,0.3)",
      color: "#fde68a",
      marginBottom: 16,
    } as React.CSSProperties,

    error: {
      padding: "14px 16px",
      borderRadius: 14,
      background: "rgba(239,68,68,0.12)",
      border: "1px solid rgba(239,68,68,0.3)",
      color: "#fecaca",
      marginBottom: 16,
    } as React.CSSProperties,

    codeBlock: {
      margin: 0,
      padding: 16,
      borderRadius: 16,
      background: "rgba(2,6,23,0.85)",
      border: "1px solid rgba(255,255,255,0.06)",
      overflowX: "auto" as const,
      color: "#cbd5e1",
      fontSize: 13,
      lineHeight: 1.6,
      whiteSpace: "pre-wrap" as const,
    } as React.CSSProperties,

    pairList: {
      display: "grid",
      gap: 14,
    } as React.CSSProperties,

    pairCard: {
      padding: 18,
      borderRadius: 16,
      background: "rgba(15,23,42,0.75)",
      border: "1px solid rgba(255,255,255,0.08)",
    } as React.CSSProperties,

    badgeRow: {
      display: "flex",
      gap: 8,
      flexWrap: "wrap" as const,
      marginBottom: 12,
    } as React.CSSProperties,

    badge: {
      display: "inline-flex",
      alignItems: "center",
      padding: "6px 10px",
      borderRadius: 999,
      fontSize: 12,
      fontWeight: 700,
      background: "rgba(59,130,246,0.14)",
      color: "#bfdbfe",
      border: "1px solid rgba(59,130,246,0.28)",
    } as React.CSSProperties,

    qaLabel: {
      fontSize: 12,
      fontWeight: 800,
      color: "#93c5fd",
      marginBottom: 6,
      textTransform: "uppercase" as const,
      letterSpacing: "0.08em",
    } as React.CSSProperties,

    qaText: {
      margin: 0,
      color: "#f8fafc",
      lineHeight: 1.7,
      fontSize: 15,
    } as React.CSSProperties,

    pairDivider: {
      height: 1,
      background: "rgba(255,255,255,0.06)",
      margin: "14px 0",
    } as React.CSSProperties,
  };

  return (
    <main style={styles.page}>
      <div style={styles.container}>
        <section style={styles.hero}>
          <h1 style={styles.title}>URL Q/A Capacity Evaluator</h1>
          <p style={styles.subtitle}>
            This Agentic app evaluate an imported URL, estimate high-quality training pairs by Regex-based text analysis, Heuristic feature scoring, Weighted rule estimation, Thresholds and Confidence scorring for quality Q and A pairs,
            and generate question-answer. This agentic system is for Researchers, RAG enthuisits, Fine tunning LLMs and students who want to extract a quality question and answer from URL/websites as extracting Q and A datas fom website manullay is a handy job and time consumable ,thus, this Agentic system try to provide ease of Q ana A data collection services.  
            To generate: 1.Put a URL 2.click Evaluate 3.click generate 4.Download. Depending on the number of Q and A pairs it will take time.    
	</p>
        </section>

        <form onSubmit={onEvaluate} style={styles.formCard}>
          <div>
            <label style={styles.label}>Target URL</label>
            <input
              type="url"
              placeholder="https://en.wikipedia.org/wiki/Electrocardiography"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              required
              style={styles.input}
            />
          </div>

          <div>
            <label style={styles.label}>Strictness</label>
            <select
              value={strictness}
              onChange={(e) => setStrictness(e.target.value as StrictnessLabel)}
              style={styles.select}
            >
              {STRICTNESS_OPTIONS.map((option) => (
                <option key={option.label} value={option.label}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div style={styles.buttonRow}>
            <button
              type="submit"
              disabled={loading}
              style={styles.primaryButton}
            >
              {evaluating ? "Evaluating..." : "Evaluate URL"}
            </button>

            {result && (
              <button
                type="button"
                onClick={onCreateJob}
                disabled={loading}
                style={styles.secondaryButton}
              >
                {generating ? "Generating..." : "Generate Quality Pairs"}
              </button>
            )}

            {jobId && (
              <a href={jobPageUrl} style={styles.linkButton}>
                Open Job Page
              </a>
            )}

            {jobId && jobStatus === "completed" && (
              <>
                <a
                  href={downloadJsonUrl}
                  target="_blank"
                  rel="noreferrer"
                  download
                  style={styles.linkButton}
                >
                  Download JSON
                </a>
                <a
                  href={downloadCsvUrl}
                  target="_blank"
                  rel="noreferrer"
                  download
                  style={styles.linkButton}
                >
                  Download CSV
                </a>
              </>
            )}
          </div>
        </form>

        {error && <div style={styles.error}>{error}</div>}

        {result && (
          <>
            <div style={styles.grid}>
              <div style={styles.metricCard}>
                <div style={styles.metricLabel}>Training-grade pairs</div>
                <div style={styles.metricValue}>{result.training_grade_pairs}</div>
              </div>

              <div style={styles.metricCard}>
                <div style={styles.metricLabel}>Raw extractable pairs</div>
                <div style={styles.metricValue}>{result.raw_extractable_pairs}</div>
              </div>

              <div style={styles.metricCard}>
                <div style={styles.metricLabel}>Predicted range</div>
                <div style={{ ...styles.metricValue, fontSize: 22 }}>
                  {result.predicted_min} - {result.predicted_max}
                </div>
              </div>

              <div style={styles.metricCard}>
                <div style={styles.metricLabel}>Confidence</div>
                <div style={{ ...styles.metricValue, fontSize: 22 }}>
                  {result.confidence}
                </div>
              </div>
            </div>

            <section style={styles.section}>
              <h2 style={styles.sectionTitle}>Evaluation Summary</h2>

              <div style={styles.infoRow}>
                <div style={styles.infoLabel}>Title</div>
                <div style={styles.infoValue}>{result.title || "—"}</div>
              </div>
              <div style={styles.infoRow}>
                <div style={styles.infoLabel}>Status</div>
                <div style={styles.infoValue}>{result.status || "—"}</div>
              </div>
              <div style={styles.infoRow}>
                <div style={styles.infoLabel}>Method</div>
                <div style={styles.infoValue}>{result.method || "—"}</div>
              </div>
              <div style={styles.infoRow}>
                <div style={styles.infoLabel}>Generation Mode</div>
                <div style={styles.infoValue}>
                  Generate exactly <strong>{result.training_grade_pairs}</strong> quality pairs
                </div>
              </div>
            </section>

            {(jobId || jobStatus || jobResult) && (
              <section style={styles.section}>
                <h2 style={styles.sectionTitle}>Job Status</h2>

                {jobStatus === "completed" && (
                  <div style={styles.success}>
                    Job completed. Showing generated quality Q&amp;A pairs below.
                  </div>
                )}

                {(jobStatus === "queued" ||
                  jobStatus === "running" ||
                  generating) && (
                  <div style={styles.warning}>
                    Job is still processing. Current status:{" "}
                    <strong>{jobStatus || "starting"}</strong>
                    {jobStep ? ` · ${jobStep}` : ""}
                    {" · "}
                    {jobProgress}%
                  </div>
                )}

                {jobId && (
                  <div style={styles.infoRow}>
                    <div style={styles.infoLabel}>Job ID</div>
                    <div style={styles.infoValue}>{jobId}</div>
                  </div>
                )}

                {jobStatus && (
                  <div style={styles.infoRow}>
                    <div style={styles.infoLabel}>Status</div>
                    <div style={styles.infoValue}>{jobStatus}</div>
                  </div>
                )}

                <div style={styles.infoRow}>
                  <div style={styles.infoLabel}>Progress</div>
                  <div style={styles.infoValue}>{jobProgress}%</div>
                </div>

                <div style={{ ...styles.infoRow, borderBottom: "none" }}>
                  <div style={styles.infoLabel}>Step</div>
                  <div style={styles.infoValue}>{jobStep || "—"}</div>
                </div>
              </section>
            )}

            {jobResult?.summary && (
              <section style={styles.section}>
                <h2 style={styles.sectionTitle}>Generated Summary</h2>
                <pre style={styles.codeBlock}>{jobResult.summary}</pre>
              </section>
            )}

            {jobResult && (
              <section style={styles.section}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 12,
                    flexWrap: "wrap",
                    marginBottom: 16,
                  }}
                >
                  <h2 style={{ ...styles.sectionTitle, marginBottom: 0 }}>
                    Job Result
                  </h2>

                  {jobId && jobStatus === "completed" && (
                    <div style={styles.buttonRow}>
                      <a
                        href={downloadJsonUrl}
                        target="_blank"
                        rel="noreferrer"
                        download
                        style={styles.linkButton}
                      >
                        Download JSON
                      </a>
                      <a
                        href={downloadCsvUrl}
                        target="_blank"
                        rel="noreferrer"
                        download
                        style={styles.linkButton}
                      >
                        Download CSV
                      </a>
                    </div>
                  )}
                </div>

                <pre style={styles.codeBlock}>
                  {JSON.stringify(jobResult, null, 2)}
                </pre>
              </section>
            )}

            {jobStatus === "completed" && generatedPairs.length > 0 && (
              <section style={styles.section}>
                <h2 style={styles.sectionTitle}>
                  Generated Q&amp;A Pairs ({generatedPairs.length})
                </h2>

                <div style={styles.pairList}>
                  {generatedPairs.map((pair, index) => (
                    <div key={index} style={styles.pairCard}>
                      <div style={styles.badgeRow}>
                        <span style={styles.badge}>Pair #{index + 1}</span>
                        {pair.type && <span style={styles.badge}>{pair.type}</span>}
                        {pair.quality_score != null && (
                          <span style={styles.badge}>Score: {pair.quality_score}</span>
                        )}
                        {pair.source && <span style={styles.badge}>{pair.source}</span>}
                      </div>

                      <div>
                        <div style={styles.qaLabel}>Question</div>
                        <p style={styles.qaText}>{pair.question}</p>
                      </div>

                      <div style={styles.pairDivider} />

                      <div>
                        <div style={styles.qaLabel}>Answer</div>
                        <p style={styles.qaText}>{pair.answer}</p>
                      </div>

                      {pair.context && (
                        <>
                          <div style={styles.pairDivider} />
                          <div>
                            <div style={styles.qaLabel}>Context</div>
                            <p style={styles.qaText}>{pair.context}</p>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section style={styles.section}>
              <h2 style={styles.sectionTitle}>Quality View</h2>
              <pre style={styles.codeBlock}>
                {JSON.stringify(result.quality_view, null, 2)}
              </pre>
            </section>

            <section style={styles.section}>
              <h2 style={styles.sectionTitle}>Metrics</h2>
              <pre style={styles.codeBlock}>
                {JSON.stringify(result.metrics, null, 2)}
              </pre>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
