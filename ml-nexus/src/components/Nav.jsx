import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { RxCross1, RxHamburgerMenu } from "react-icons/rx";

function Nav() {
  const [isOpen, setIsOpen] = useState(false);

  // Function to toggle the mobile menu
  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="w-full h-20 p-8 flex items-center justify-between border-b border-b-slate-100 bg-[#324655]">
      {/* Logo */}
      <h1 className="text-xl font-bold tracking-tight text-white ml-4">ML Nexus</h1>

      {/* Desktop Navigation Links */}
      <div className="hidden md:flex gap-10">
        <Link className='text-white hover:text-[#61B3A0]' to="/">Home</Link>
        <Link className='text-white hover:text-[#61B3A0]' to="/about">About</Link>
        <Link className='text-white hover:text-[#61B3A0]' to="/contact">Contact</Link>
      </div>

      {/* Hamburger Menu Button */}
      <button
        onClick={toggleMenu}
        className="md:hidden transition duration-300 ease-in-out focus:outline-none text-xl text-white mr-4"
      >
        {isOpen ? <RxCross1 /> : <RxHamburgerMenu />}
      </button>

      {/* Mobile Menu - Shows when isOpen is true */}
      {isOpen && (
        <div className="absolute top-16 left-0 w-full bg-[#324655] border-t border-slate-100 flex flex-col items-center py-5 gap-4 md:hidden">
          <Link className='text-white hover:text-[#61B3A0]' to="/" onClick={toggleMenu}>Home</Link>
          <Link className='text-white hover:text-[#61B3A0]' to="/About" onClick={toggleMenu}>About</Link>
          <Link className='text-white hover:text-[#61B3A0]' to="/Contact" onClick={toggleMenu}>Contact</Link>
        </div>
      )}
    </div>
  );
}

export default Nav;
