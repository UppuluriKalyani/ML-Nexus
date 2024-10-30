// // import React, { useEffect, useState } from 'react';
// // import './CSS/Random.css';

// // const Random = () => {
// //   const [visible, setVisible] = useState(false);

// //   useEffect(() => {
// //     const timer = setTimeout(() => {
// //       setVisible(true);
// //     }, 100); // 1-second delay

// //     return () => clearTimeout(timer);
// //   }, []);

// //   return (
// //     <div className={`random-container ${visible ? 'visible' : ''}`}>
// //       <h2 className="random-title">Welcome to the Research Assistant</h2>
// //       <p className="random-paragraph">
// //         This tool helps you extract information from research papers and provides tools to critically analyze and apply them to your writing. Sign up now and try it for free!
// //       </p>
      
// //         <strong>Try for free</strong>
      
// //     </div>
// //   );
// // };

// // export default Random;

// import React, { useEffect, useState } from 'react';
// import './CSS/Random.css';

// const Random = () => {
//   const [visible, setVisible] = useState(false);
//   const [showFirstContent, setShowFirstContent] = useState(true);

//   useEffect(() => {
//     const timer = setTimeout(() => {
//       setVisible(true);
//     }, 1000); // 1-second delay

//     const interval = setInterval(() => {
//       setShowFirstContent(prev => !prev);
//     }, 6000); // Switch content every 6 seconds

//     return () => {
//       clearTimeout(timer);
//       clearInterval(interval);
//     };
//   }, []);

//   return (
//     <div className={`random-container ${visible ? 'visible' : ''}`}>
//       {showFirstContent ? (
//         <div className="content-block">
//           <h2 className="random-title">Welcome to the Research Assistant</h2>
//           <p className="random-paragraph">
//             This tool helps you extract information from research papers and provides tools to critically analyze and apply them to your writing. Sign up now and try it for free!
//           </p>
//         </div>
//       ) : (
//         <div className="content-block">
//           <h2 className="random-title">Another Important Feature</h2>
//           <p className="random-paragraph">
//             Our research assistant also helps you with automated summarization, citation management, and much more. Explore all features today!
//           </p>
//         </div>
//       )}
      
        
      
//     </div>
//   );
// };

// export default Random;

import React, { useEffect, useRef, useState } from 'react';
import './CSS/Random.css';

const Random = () => {
  const blockRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsVisible(entry.isIntersecting);
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
      }
    );

    if (blockRef.current) {
      observer.observe(blockRef.current);
    }

    return () => {
      if (blockRef.current) {
        observer.unobserve(blockRef.current);
      }
    };
  }, []);

  return (
    <div ref={blockRef} className={`random-container ${isVisible ? 'visible' : ''}`}>
      <div className="content-block">
        <h2 className="random-title">Welcome to the Research Assistant</h2>
        <p className="random-paragraph">
          This tool helps you extract information from research papers and provides tools to critically analyze and apply them to your writing. Sign up now and try it for free!
        </p>
        
      </div>
    </div>
  );
};

export default Random;
