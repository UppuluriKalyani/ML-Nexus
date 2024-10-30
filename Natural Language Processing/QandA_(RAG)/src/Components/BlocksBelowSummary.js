import React, { useEffect, useRef, useState } from 'react';
import Footer from '../Components/Footer.js';
import './CSS/BlocksBelowSummary.css';

const BlocksBelowSummary = () => {
  const blockRefs = useRef([]);
  const [visibleBlocks, setVisibleBlocks] = useState([false, false, false]);

  useEffect(() => {
    if (!('IntersectionObserver' in window)) {
      setVisibleBlocks([true, true, true]);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry, index) => {
          if (entry.isIntersecting) {
            setTimeout(() => {
              setVisibleBlocks((prevVisibleBlocks) => {
                const newVisibleBlocks = [...prevVisibleBlocks];
                newVisibleBlocks[index] = true;
                return newVisibleBlocks;
              });
            }, (index + 1) * 500); // Delay appearance of each block
          }
        });
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.1,
      }
    );

    blockRefs.current.forEach((block) => {
      if (block) {
        observer.observe(block);
      }
    });

    return () => {
      blockRefs.current.forEach((block) => {
        if (block) {
          observer.unobserve(block);
        }
      });
    };
  }, []);

  return (
    <>
      <div className="blocks-container" style={{marginTop:'80px'}}>
        {['I think you got your summary correct', 'Have any doubts about your Research Paper', 'Ask your questions below here'].map((title, index) => (
          <div
            key={index}
            ref={(el) => (blockRefs.current[index] = el)}
            className={`random-container ${visibleBlocks[index] ? 'visible' : ''}`}
          >
            <div className="content-block">
              <h2 className="random-title">{title}</h2>
              <p className="random-paragraph">
                This tool helps you extract information from research papers and provides tools to critically analyze and apply them to your writing. Sign up now and try it for free!
              </p>
            </div>
          </div>
        ))}
      </div>
      
    </>
  );
};

export default BlocksBelowSummary;
