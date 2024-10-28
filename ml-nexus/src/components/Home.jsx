import React from 'react'
import Nav from './Nav'

function Home() {
  return (
    <div className='h-screen dark:bg-[#253341] w-full dark:text-[#AFC2CB]'>
      <Nav />
      <div className="h-full w-full flex justify-center">
      <div className=" flex mx-auto flex-col items-center">
      <h1 className='text-2xl mx-auto mt-20 text-center'> Welcome To <br /> <span className='dark:text-green-400 inline-block text-9xl'>ML Nexus<span className='text-[#AFC2CB]'>.</span> </span></h1>
      <p className='w-[65vw] text-center mt-6'>This repository is dedicated to providing top-quality machine learning tools and resources. Track our milestones, see top programming languages in use, and monitor community progress—all in one place.</p>
      <p className='mt-6'>Your gateway to collaborative machine learning projects, hands-on experiments, and AI innovations.</p>
      </div>
      </div>
    </div>
  )
}

export default Home