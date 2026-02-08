import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [city, setCity] = useState("");
  const [cities, setCities] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/cities`)
      .then((res) => res.json())
      .then((data) => setCities(data.cities || []))
      .catch(() => setCities([]));
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!city.trim()) {
      setError("Please enter a city name.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(
        `${API_URL}/predict?city=${encodeURIComponent(city.trim())}&live=true`,
      );

      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg.detail || "Failed to fetch prediction");
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Urban Air Intelligence</p>
          <h1>City Pulse</h1>
          <p className="subtitle">
            Enter a city to see NO2 and related pollutant signals.
          </p>
        </div>
      </header>

      <main className="panel">
        <form className="search" onSubmit={handleSubmit}>
          <label htmlFor="city">City name</label>
          <div className="search-row">
            <select value={city} onChange={(e) => setCity(e.target.value)}>
              <option value="">Select a city</option>
              {cities.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <button type="submit" disabled={loading}>
              {loading ? "Loading..." : "Get Air Quality"}
            </button>
          </div>
        </form>

        {error && <div className="notice error">{error}</div>}

        {result && (
          <section className="results">
            <div className="results-header">
              <div>
                <h2>{result.city}</h2>
                <p>{result.date}</p>
              </div>
              <div className="coords">
                {result.latitude.toFixed(4)}, {result.longitude.toFixed(4)}
              </div>
            </div>

            <div className="grid">
              <div className="card">
                <h3>MODIS AOD</h3>
                <p>
                  {Number.isFinite(result.modis_aod)
                    ? result.modis_aod.toFixed(3)
                    : "Unavailable"}
                </p>
                {result.modis_granule_id && (
                  <p className="muted">Granule {result.modis_granule_id}</p>
                )}
                {result.modis_error && (
                  <p className="muted">{result.modis_error}</p>
                )}
              </div>
              <div className="card">
                <h3>NO2 Average</h3>
                <p>
                  {Number.isFinite(result.no2_avg)
                    ? result.no2_avg.toFixed(3)
                    : "Unavailable"}
                </p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
