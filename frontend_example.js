
import React, { useState, useEffect } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithEmailAndPassword, onAuthStateChanged } from 'firebase/auth';

// --- 1. Firebase Configuration ---
// IMPORTANT: Replace with your Firebase project's configuration
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// --- 2. React Component ---
const ApiClient = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [user, setUser] = useState(null);
  const [apiKey, setApiKey] = useState(null);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  const handleLogin = async () => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogout = async () => {
    await auth.signOut();
    setApiKey(null);
    setQueryResult(null);
  };

  const handleGenerateApiKey = async () => {
    if (!user) {
      setError('You must be logged in to generate an API key.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const token = await user.getIdToken();
      const response = await fetch('http://localhost:8000/generate-key', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to generate API key');
      }

      const data = await response.json();
      setApiKey(data.data.api_key);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!apiKey) {
      setError('You must have an API key to query the API.');
      return;
    }

    setLoading(true);
    setError(null);
    setQueryResult(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({ query: query, max_results: 5 })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to query');
      }

      const data = await response.json();
      setQueryResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>RAG API Client</h1>
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {!user ? (
        <div>
          <h2>Login</h2>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
          <button onClick={handleLogin}>Login</button>
        </div>
      ) : (
        <div>
          <p>Welcome, {user.email}</p>
          <button onClick={handleLogout}>Logout</button>
          <hr />

          {!apiKey ? (
            <div>
              <button onClick={handleGenerateApiKey} disabled={loading}>
                {loading ? 'Generating...' : 'Generate API Key'}
              </button>
            </div>
          ) : (
            <div>
              <p>Your API Key: <code>{apiKey}</code></p>
              <hr />
              <h2>Query the API</h2>
              <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Enter your query" />
              <button onClick={handleQuery} disabled={loading}>
                {loading ? 'Querying...' : 'Submit Query'}
              </button>

              {queryResult && (
                <div>
                  <h3>Answer:</h3>
                  <p>{queryResult.answer}</p>
                  <h3>Sources:</h3>
                  <ul>
                    {queryResult.sources.map((source, index) => (
                      <li key={index}>{source.metadata.filename} (Score: {source.similarity_score.toFixed(4)})</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ApiClient;
