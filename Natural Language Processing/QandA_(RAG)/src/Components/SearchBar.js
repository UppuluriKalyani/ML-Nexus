
// import React, { useState } from 'react';
// import './CSS/SearchBar.css';
// import AnswerBlock from './AnswerBlock'; // Import the AnswerBlock component

// const SearchBar = ({ onSearch }) => {
//   const [query, setQuery] = useState('');
//   const [error, setError] = useState('');
//   const [answer, setAnswer] = useState('');

//   const handleInputChange = (e) => {
//     setQuery(e.target.value);
//   };

//   const handleSearch = () => {
//     if (query.trim() !== '') {
//       fetch('http://localhost:5000/ask', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ query: query }),
//       })
//         .then(response => {
//           if (!response.ok) {
//             throw new Error('Network response was not ok');
//           }
//           return response.json();
//         })
//         .then(data => {
//           console.log(data); // Log the response to inspect
//           setAnswer(data.answer); // Store the answer
//           onSearch(data.answer); // Pass the answer to the parent component
//         })
//         .catch(error => {
//           console.error('Error:', error);
//           setError(error);
//         });
//     } else {
//       setError('Please enter a query');
//     }
//   };

//   return (
//     <>
//       <div>
//         <div className="search-bar" style={{ marginTop: '120px', alignItems: 'center' }}>
//           <input
//             type="text"
//             value={query}
//             onChange={handleInputChange}
//             placeholder="Ask a question about the research paper..."
//           />
//           <button onClick={handleSearch}>Search</button>
//         </div>
//         <div style={{ alignItems: 'center' }}>
//           <h1>The answers for the asked questions will display here</h1>
//           <div className="result-container" style={{ alignItems: 'center' }}>
//             {answer && <AnswerBlock answer={answer} />} {/* Use the AnswerBlock component */}
//           </div>
//         </div>
//       </div>
//     </>
//   );
// };

// export default SearchBar;



import React, { useState } from 'react';
import './CSS/SearchBar.css';
import AnswerBlock from './AnswerBlock'; // Import the AnswerBlock component

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false); // Add loading state

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSearch = () => {
    if (query.trim() !== '') {
      setLoading(true); // Start loading animation
      setAnswer(''); // Clear the previous answer
      fetch('http://localhost:5000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
      })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          console.log(data); // Log the response to inspect
          setAnswer(data.answer); // Store the answer
          setLoading(false); // Stop loading animation
          onSearch(data.answer); // Pass the answer to the parent component
        })
        .catch(error => {
          console.error('Error:', error);
          setError(error);
          setLoading(false); // Stop loading animation in case of error
        });
    } else {
      setError('Please enter a query');
    }
  };

  return (
    <>
      <div>
        <div className="search-bar" style={{ marginTop: '120px', alignItems: 'center' }}>
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            placeholder="Ask a question about the research paper..."
          />
          <button onClick={handleSearch}>Search</button>
        </div>
        <div style={{ alignItems: 'center' }}>
          <h1>The answers for the asked questions will display here</h1>
          <div className="result-container" style={{ alignItems: 'center' }}>
            {loading && <div className="loading-spinner"></div>}
            {answer && !loading && <AnswerBlock answer={answer} />} {/* Use the AnswerBlock component */}
          </div>
        </div>
      </div>
    </>
  );
};

export default SearchBar;
