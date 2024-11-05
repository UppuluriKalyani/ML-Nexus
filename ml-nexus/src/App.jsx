import React, { useEffect } from 'react';
import LocomotiveScroll from 'locomotive-scroll'; // Make sure locomotive-scroll is installed
import 'locomotive-scroll/dist/locomotive-scroll.css'; // Import the styles

import Nav from './components/Nav';
import Home from './components/Home';
import Routing from './utils/routing'; // Assuming you renamed your component to Routing

function App() {
  useEffect(() => {
    // Initialize LocomotiveScroll
    const scroll = new LocomotiveScroll({
      el: document.querySelector('.scroll-container'), // The container where scroll should be applied
      smooth: true, // Enables smooth scrolling
      getDirection: true, // Tracks the scroll direction
      lerp: 0.1, // Sets the smooth scroll rate (adjust as needed)
    });

    // Cleanup LocomotiveScroll on unmount
    return () => scroll.destroy();
  }, []);

  return (
    <div className="h-full w-full bg-[#E8F0F2] scroll-container">
      <Nav />
      <Routing /> {/* Render the Routing component for navigation */}
    </div>
  );
}

export default App;
