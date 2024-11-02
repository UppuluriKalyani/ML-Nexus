import React, { useContext, useEffect } from 'react'
import Btn from './Btn'
import {repoContext} from '../utils/Context'
import { motion } from 'framer-motion'
function Projects() {

    const {projects} = useContext(repoContext)

  
  return (
    <motion.div whileHover={{padding: "25px"}} className="p-10">
        <h1 className='text-4xl font-bold -tracking-normal leading-none mb-6'>Projects</h1>
        <div className="flex flex-wrap gap-7 items-center justify-center">
      {projects.map((p,i) => (
          <motion.div whileHover={{padding: "30px", backgroundColor: "#3A5064"}} key={i} className="py-5 px-5 hover:text-[#B9D1DA] h-40 w-42 md:h-56 md:w-80 bg-[#324655] rounded-lg flex flex-col justify-between items-center">
          <h1 className='text-3xl'>{p.name}</h1>
          <Btn className=' rounded-md hover:bg-red-100 ' value={{name:"View Models", ref: p.html_url }} />
      </motion.div>
      ))}
        </div>
    </motion.div>
  )
}

export default Projects