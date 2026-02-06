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
        `${API_URL}/predict?city=${encodeURIComponent(city.trim())}`,
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
            <input
              id="city"
              list="cities"
              placeholder="e.g., Delhi"
              value={city}
              onChange={(e) => setCity(e.target.value)}
            />
            <datalist id="cities">
              {cities.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
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
                <h3>NO2 Avg</h3>
                <p>{result.no2_avg.toFixed(6)}</p>
              </div>
              <div className="card">
                <h3>NO2 Predicted</h3>
                <p>{result.no2_predicted.toFixed(6)}</p>
              </div>
              <div className="card">
                <h3>PM2.5</h3>
                <p>{result.pm25.toFixed(2)}</p>
              </div>
              <div className="card">
                <h3>PM10</h3>
                <p>{result.pm10.toFixed(2)}</p>
              </div>
              <div className="card">
                <h3>SO2</h3>
                <p>{result.so2.toFixed(2)}</p>
              </div>
              <div className="card">
                <h3>CO</h3>
                <p>{result.co.toFixed(2)}</p>
              </div>
              <div className="card">
                <h3>O3</h3>
                <p>{result.o3.toFixed(2)}</p>
              </div>
              <div className="card">
                <h3>Weather</h3>
                <p>
                  {result.temperature}C, {result.humidity}%
                </p>
                <p>Wind {result.wind_speed} m/s</p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
