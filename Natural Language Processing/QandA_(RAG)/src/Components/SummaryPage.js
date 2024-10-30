// import React, { useEffect, useState } from 'react';
// import BlocksBelowSummary from './BlocksBelowSummary';
// import SearchBar from './SearchBar';
// import './CSS/SummaryPage.css';
// import Footer from './Footer';
// import { useLocation } from 'react-router-dom';

// const SummaryPage = () => {
//     const [summary, setSummary] = useState(''); // State to store the summary
//     const [loading, setLoading] = useState(true); // State for loading animation
//     const location = useLocation();

//     useEffect(() => {
//         window.scrollTo(0, 0);
//         // Get the summary from location state
//         if (location.state && location.state.summary) {
//             setSummary(location.state.summary);
//         }
//         setLoading(false); // Stop loading animation once summary is set
//     }, [location]);

//     const handleSearch = (query) => {
//         // Handle the search query
//         console.log('Search query:', query);
//         // You can send this query to your backend to get the related answers
//     };

//     return (
//         <>
//             <div className='header-margin'>
//                 <h1>Your Summary will be displayed here below</h1>
//             </div>
//             <div className="summary-page-container">
//                 {loading ? (
//                     <div className="loading-spinner"></div>
//                 ) : (
//                     <div className="rectangle-box">
//                         <div className="text-aligning">
//                             <p>{summary}</p>
//                         </div>
//                     </div>
//                 )}
//                 <button className="mic-button">🎤</button>
//             </div>
//             <BlocksBelowSummary />
//             <SearchBar onSearch={handleSearch} />
//             <h1>The answers for the asked questions should display here</h1>
//             <Footer />
//         </>
//     );
// };

// export default SummaryPage;


import React, { useEffect, useState } from 'react';
import BlocksBelowSummary from './BlocksBelowSummary';
import SearchBar from './SearchBar';
import './CSS/SummaryPage.css';
import Footer from './Footer';
import { useLocation } from 'react-router-dom';

const SummaryPage = () => {
    const [summary, setSummary] = useState(''); // State to store the summary
    const [loading, setLoading] = useState(true); // State for loading animation
    const location = useLocation();

    useEffect(() => {
        window.scrollTo(0, 0);
        // Get the summary from location state
        if (location.state && location.state.summary) {
            console.log("Summary from location state:", location.state.summary); // Debug logging
            setSummary(location.state.summary);
        } else {
            console.log("No summary found in location state."); // Debug logging
        }
        setLoading(false); // Stop loading animation once summary is set
    }, [location]);

    const handleSearch = (query) => {
        // Handle the search query
        console.log('Search query:', query);
        // You can send this query to your backend to get the related answers
    };

    return (
        <>
            <div className='header-margin'>
                <h1>Your Summary will be displayed here below</h1>
            </div>
            <div className="summary-page-container">
                {loading ? (
                    <div className="loading-spinner"></div>
                ) : (
                    <div className="rectangle-box">
                        <div className="text-aligning">
                            <p>{summary}</p>
                        </div>
                    </div>
                )}
                <button className="mic-button">🎤</button>
            </div>
            <BlocksBelowSummary />
            <SearchBar onSearch={handleSearch} />
            
            <Footer />
        </>
    );
};

export default SummaryPage;
