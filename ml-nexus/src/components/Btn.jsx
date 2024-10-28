import React from 'react'
import { FaGithub } from "react-icons/fa";
function Btn({value, git=false }) {
  return (
    <button className=' flex items-center py-2 px-5 rounded-full dark:bg-[#61B3A0] dark:text-[#28333F] gap-2 '>
      {git && <FaGithub />}
      {value}
    </button>
  )
}

export default Btn