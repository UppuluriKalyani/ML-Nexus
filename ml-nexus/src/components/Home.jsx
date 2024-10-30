import React from 'react'
import Nav from './Nav'
import Btn from './Btn'
import Hero from './Hero'
import Projects from './Projects'
import Statistics from './Statistics'
import Contributors from './Contributors'

function Home() {

  return (
    <div className='h-full overflow-hidden dark:bg-[#253341] w-full dark:text-[#AFC2CB]'>
      <Nav />
      <Hero />
      <Projects />
      <Statistics />
      <Contributors />
    </div>
  )
}

export default Home