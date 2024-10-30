// import React from 'react';
// import './CSS/BlocksContent.css';

// const BlocksContent = () => {
//   return (
//     <div className="paper-content1 research-paper1 browser__frame1 xl">
//       <div className="hero-content">
        
//         <div style={{ opacity: 1 }} className="tooltop">
//           <div className="head-tooltip">
//             <img
//               src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/660170a3e43bbbfbf4143b9f_researcher__icn.svg"
//               loading="lazy"
//               alt=""
//               className="tool-tip-image"
//             />
//             <h3 className="tool-tip-h3">Your trusty research assistant</h3>
//           </div>
//           <div className="arrow-down"></div>
//         </div>
//         <div
//           style={{
//             opacity: 1,
//             transform:
//               'translate3d(0px, 0px, 0px) scale3d(1, 1, 1) rotateX(0deg) rotateY(0deg) rotateZ(0deg) skew(0deg, 0deg)',
//             transformStyle: 'preserve-3d'
//           }}
//           className="paper-hero p-relative z-index-5"
//         >
//           <p className="p-large">
//             <span className="text-span-2">Helps you </span>
//             <span className="bg-light-yellow">extract information from that pile of research papers</span>
//             <span className="text-span-2">
//               {' '}
//               and gives you the tools to critically analyse and apply them to your writing.
//             </span>
//           </p>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default BlocksContent;


import React, { useEffect, useState } from 'react';
import './CSS/BlocksContent.css';

const BlocksContent = () => {
  const [visible, setVisible] = useState(false);
  const [currentInfo, setCurrentInfo] = useState(0);
  const information = [
    {
      title: 'Your trusty research assistant',
      description: 'Closed domain approach increases relevance of the answers.'
    }
  ];

  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 1000); // 1-second delay

    const interval = setInterval(() => {
      setCurrentInfo(prev => (prev + 1) % information.length);
    }, 6000); // Switch content every 6 seconds

    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className={`paper-content1 research-paper1 browser__frame1 xl ${visible ? 'visible' : ''}`}>
      <div className="hero-content">
        <div className="tooltop">
          <div className="head-tooltip">
            <img
              src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/660170a3e43bbbfbf4143b9f_researcher__icn.svg"
              loading="lazy"
              alt=""
              className="tool-tip-image"
            />
            <h3 className="tool-tip-h3">{information[currentInfo].title}</h3>
          </div>
          <div className="arrow-down"></div>
        </div>
        <div className="paper-hero p-relative z-index-5">
          <p className="p-large">
            {information[currentInfo].description}
          </p>
        </div>
      </div>
    </div>
  );
};

export default BlocksContent;
