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
    <div className="h-20 max-w-screen-lg mx-auto flex py-10 px-2 items-center justify-between border-b border-b-slate-100">
      {/* Logo */}
      <h1 className="text-xl font-bold tracking-tight">ML Nexus</h1>

      {/* Desktop Navigation Links */}
      <div className="hidden md:flex gap-10">
        <Link className='hover:text-[#61B3A0]' to="/">Home</Link>
        <Link className='hover:text-[#61B3A0]' to="/about">About</Link>
        <Link className='hover:text-[#61B3A0]' to="/contact">Contact</Link>
      </div>

      {/* Hamburger Menu Button */}
      <button
        onClick={toggleMenu}
        className="md:hidden transition duration-300 ease-in-out focus:outline-none text-xl"
      >
        {isOpen ? <RxCross1 /> : <RxHamburgerMenu />}
      </button>

      {/* Mobile Menu - Shows when isOpen is true */}
      {isOpen && (
        <div className="absolute top-16 left-0 w-full dark:bg-[#324655] border-t border-slate-100 flex flex-col items-center py-5 px-5 gap-4 md:hidden">
          <Link className='hover:text-[#61B3A0]' to="/" onClick={toggleMenu}>Home</Link>
          <Link className='hover:text-[#61B3A0]' to="/About" onClick={toggleMenu}>About</Link>
          <Link className='hover:text-[#61B3A0]' to="/Contact" onClick={toggleMenu}>Contact</Link>
        </div>
      )}
    </div>
  );
}

export default Nav;
