import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';

// Import global styles for the application
import './index.css';

// The main App component, typed with React.FC (Functional Component)
const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          {/* The primary route will render the Trading Terminal Dashboard */}
          <Route path="/" element={<Dashboard />} />
          
          {/* Placeholder routes for future pages based on our UI blueprint */}
          {/* <Route path="/portfolio" element={<div>Portfolio Page</div>} /> */}
          {/* <Route path="/intelligence" element={<div>Market Intelligence Page</div>} /> */}
          {/* <Route path="/system" element={<div>System Health Page</div>} /> */}
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
