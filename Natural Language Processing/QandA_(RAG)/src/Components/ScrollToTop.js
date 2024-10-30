// src/components/ScrollToTop.js
import { useEffect } from 'react';

const ScrollToTop = () => {
  useEffect(() => {
    const handleScrollToTop = () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
    };

    const scrollToTopButton = document.querySelector('.js-return-top');
    if (scrollToTopButton) {
      scrollToTopButton.addEventListener('click', handleScrollToTop);
    }

    return () => {
      if (scrollToTopButton) {
        scrollToTopButton.removeEventListener('click', handleScrollToTop);
      }
    };
  }, []);

  return null;
};

export default ScrollToTop;
