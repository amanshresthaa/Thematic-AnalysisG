import React, { useState } from 'react';

function App() {
  const [responseData, setResponseData] = useState('');

  const handleButtonClick = async () => {
    try {
      // Call the FastAPI endpoint
      const response = await fetch('http://localhost:8000/run-script');
      const data = await response.json();
      setResponseData(JSON.stringify(data, null, 2));
    } catch (error) {
      console.error(error);
      setResponseData('Error occurred while calling the API');
    }
  };

  return (
    <div className="App" style={{ margin: "20px" }}>
      <h2>React + FastAPI Demo</h2>
      <button onClick={handleButtonClick}>
        Run src/main.py
      </button>
      <pre>{responseData}</pre>
    </div>
  );
}

export default App;
