import {
  getJob,
  getJobResults,
  getJobResultsJsonUrl,
  getJobResultsCsvUrl,
} from "../../../lib/api";

export default async function JobPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const job = await getJob(id);
  const results = await getJobResults(id);

  const acceptedPairs = results.accepted_pairs || [];

  return (
    <div className="container">
      <div className="card">
        <h1>Job {id}</h1>
        <p>Status: {job.status}</p>
        <p>Progress: {job.progress}%</p>
        <p>Step: {job.step}</p>
        {job.error_message && (
          <p style={{ color: "#ff8c8c" }}>{job.error_message}</p>
        )}

        <div style={{ display: "flex", gap: "12px", marginTop: "16px" }}>
          <a
            href={getJobResultsJsonUrl(id)}
            target="_blank"
            rel="noreferrer"
            className="button"
          >
            Download JSON
          </a>
          <a
            href={getJobResultsCsvUrl(id)}
            target="_blank"
            rel="noreferrer"
            className="button"
          >
            Download CSV
          </a>
        </div>
      </div>

      <div className="card">
        <h2>Summary</h2>
        <pre>{results.summary || "No summary yet"}</pre>
      </div>

      <div className="card">
        <h2>Capacity</h2>
        <pre>{JSON.stringify(results.capacity, null, 2)}</pre>
      </div>

      <div className="card">
        <h2>Metrics</h2>
        <pre>{JSON.stringify(results.metrics, null, 2)}</pre>
      </div>

      <div className="card">
        <h2>Accepted Pairs</h2>
        {acceptedPairs.length === 0 ? (
          <p>No accepted pairs yet.</p>
        ) : (
          <div style={{ display: "grid", gap: "16px" }}>
            {acceptedPairs.map((pair: any, index: number) => (
              <div
                key={index}
                style={{
                  border: "1px solid #333",
                  borderRadius: "8px",
                  padding: "12px",
                }}
              >
                <p>
                  <strong>Question:</strong> {pair.question || "—"}
                </p>
                <p>
                  <strong>Answer:</strong> {pair.answer || "—"}
                </p>
                {pair.context && (
                  <p>
                    <strong>Context:</strong> {pair.context}
                  </p>
                )}
                {pair.source && (
                  <p>
                    <strong>Source:</strong> {pair.source}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Artifacts</h2>
        <pre>{JSON.stringify(results.artifacts, null, 2)}</pre>
      </div>
    </div>
  );
}