// src/App.js
import React, { useState, useEffect } from 'react'; // useState와 useEffect를 추가로 가져옴
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import MainPage from './pages/MainPage';
import RegisterPage from './pages/RegisterPage';
import DangerPage from './pages/DangerPage';
import ListPage from './pages/ListPage';
import './App.css';

function App() {
  const [results, setResults] = useState([]); // 모든 결과를 저장할 상태
  const [dangerousPersons, setDangerousPersons] = useState([]);

  useEffect(() => {
    // 로컬 스토리지에서 위험인물 데이터를 가져옴
    const savedDangerousPersons = localStorage.getItem('dangerousPersons');
    if (savedDangerousPersons) {
      setDangerousPersons(JSON.parse(savedDangerousPersons));
    }

    // 로컬 스토리지에서 결과 데이터를 가져옴
    const savedResults = localStorage.getItem('results');
    if (savedResults) {
      setResults(JSON.parse(savedResults));
    }
  }, []);

  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route 
            path="/" 
            element={
              <MainPage 
                results={results} 
                setResults={setResults} 
                dangerousPersons={dangerousPersons} 
              />
            } 
          />
          <Route 
            path="/register" 
            element={
              <RegisterPage 
                dangerousPersons={dangerousPersons} 
                setDangerousPersons={setDangerousPersons} 
              />
            } 
          />
          <Route 
            path="/danger" 
            element={
              <DangerPage 
                dangerousPersons={dangerousPersons} 
                setDangerousPersons={setDangerousPersons} 
              />
            } 
          />
          <Route 
            path="/list" 
            element={
              <ListPage 
                dangerousPersons={dangerousPersons} 
                setDangerousPersons={setDangerousPersons} 
              />
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
