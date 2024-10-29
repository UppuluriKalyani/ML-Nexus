import React, { useContext, useEffect } from 'react'
import Btn from './Btn'
import {repoContext} from '../utils/Context'
function Projects() {

    const {projects} = useContext(repoContext)

    // useEffect(() => {
       
    // })

  return (
    <div className=" p-10">
        <h1 className='text-4xl font-bold -tracking-normal leading-none mb-6'>Projects</h1>
        <div className="flex flex-wrap gap-7 items-center justify-center">
      {projects.map((p,i) => (
          <div key={i} className="py-5 px-5 h-56 w-80 bg-[#324655] rounded-lg flex flex-col justify-between items-center">
          <h1 className='text-3xl'>{p.name}</h1>
          <Btn className=' rounded-md ' value={"View Models"} />
      </div>
      ))}
        </div>
    </div>
  )
}

export default Projects