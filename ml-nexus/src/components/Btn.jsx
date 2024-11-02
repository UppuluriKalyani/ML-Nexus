import React from 'react'
import { FaGithub } from "react-icons/fa";
import {easeOut, motion} from 'framer-motion'
function Btn({value, git=false, className='', ...props }) {
  return (
    <a target='_blank' href={value.ref} className={`flex hover:bg-red-900 font-semibold pointer-events-auto md:text-base text-xs items-center py-2 px-5 rounded-full dark:bg-[#61B3A0] dark:text-[#28333F] gap-2 ${className} `} {...props} >
      {git && <FaGithub />}
     {value.name}
    </a>
  )
}

export default Btn