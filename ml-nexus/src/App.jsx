import React from 'react'
import Nav from './components/Nav'
import Home from './components/Home'
import LocomotiveScroll from 'locomotive-scroll';

function App() {
const locomotiveScroll = new LocomotiveScroll();
  return (
    <div className=" h-full w-full bg-[#E8F0F2] ">
      <Home  />
    </div>
  )
}

export default App