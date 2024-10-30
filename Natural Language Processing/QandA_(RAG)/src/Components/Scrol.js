// src/components/ScrollMotion.js
import React, { useEffect } from 'react';
import './CSS/Scroll.css'


const Scrol = () => {
  useEffect(() => {
    const handleScroll = () => {
      const movingElement = document.getElementById('moving-element');
      const scrollPosition = window.scrollY;

      const translateX = scrollPosition * 0;
      const translateZ = scrollPosition * 0.5;

      movingElement.style.transform = `translate3d(${translateX}px, ${translateZ}px, 0)`;
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <div className="container">
      <img
        src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/66004972685e4750b31f442d_tag__summarize.svg"
        alt="Summarize Icon"
        className="apply__icn"
        id="moving-element"
      />
      <div className="content">
        <h1>Scroll Down to See the Effect</h1>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit...</p>
        <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore...</p>
        {/* Add more content to make the page scrollable */}
      </div>
    </div>
  );
};

export default Scrol;
