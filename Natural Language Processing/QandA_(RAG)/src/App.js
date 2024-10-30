import React, { useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import axios from 'axios';
import './App.css';
import MainPage from './Components/MainPage';
import MyNav from './Components/MyNav';
import SummaryPage from './Components/SummaryPage';
import AboutUs from './Components/AboutUs';
import Feedback from './Components/Feedback';
import Features from './Components/Features';
import SearchBar from './Components/SearchBar';
import FileUpload from './Components/FileUpload'; 


const App = () => {
  const [answer, setAnswer] = useState('');

  const handleSearch = async (query) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/ask', { query });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error fetching the answer:', error);
    }
  };
  return (
    <>
      <div className="App">
        <MyNav />
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/summary" element={<SummaryPage />} />
          <Route path="/features" element={<Features />} />
          <Route path="/about" element={<AboutUs />} />
          <Route path="/feedback" element={<Feedback />} />
          <Route path="/upload" element={<FileUpload />} /> 
        </Routes>
      </div>
    </>
  );
}

export default App;