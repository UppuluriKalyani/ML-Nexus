import React from 'react';
import { FaGithub, FaLinkedin } from 'react-icons/fa';
import { MdEmail } from "react-icons/md";
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-[#D6EAE8] dark:bg-[#1F2A33] text-[#28333F] dark:text-[#AFC2CB] py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center">

          <div className="text-xl font-semibold mb-4 md:mb-0">
            ML Nexus
          </div>

          <div className="flex space-x-6 text-sm md:text-base mb-4 md:mb-0">
            <Link className='text-white hover:text-[#61B3A0]' to="/about">About</Link>
            <Link className='text-white hover:text-[#61B3A0]' to="/contact">Contact</Link>
            <a href="#projects" className="hover:text-[#61B3A0]">Projects</a>
            <a href="#contributors" className="hover:text-[#61B3A0]">Contributors</a>
          </div>

          <div className="flex space-x-4">
            <a href="https://github.com/UppuluriKalyani" target="_blank" rel="noopener noreferrer" className="hover:text-[#61B3A0]">
              <i className="text-lg"> <FaGithub /> </i>
            </a>
            <a href="mailto:kalyaniuppuluri7@gmail.com" target="_blank" rel="noopener noreferrer" className="hover:text-[#61B3A0]">
              <i className="text-lg"> <MdEmail /> </i>
            </a>
            <a href="https://www.linkedin.com/in/kalyani-uppuluri-522419257/" target="_blank" rel="noopener noreferrer" className="hover:text-[#61B3A0]">
              <i className=" text-lg"><FaLinkedin /> </i>
            </a>
          </div>
        </div>

        <div className="mt-8 text-center text-sm">
          &copy; {new Date().getFullYear()} ML Nexus. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
