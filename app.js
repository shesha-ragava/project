import { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async () => {
    const res = await axios.post('http://127.0.0.1:5000/predict', { text });
    setResult(res.data.prediction);
  };

  return (
    <div style={{ padding: 40 }}>
      <h2>Stock Sentiment Analyzer</h2>
      <textarea rows="5" value={text} onChange={(e) => setText(e.target.value)} />
      <br />
      <button onClick={handleSubmit}>Analyze</button>
      {result && <h3>Result: {result}</h3>}
    </div>
  );
}

export default App;
