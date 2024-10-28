import React from 'react'
import { Link } from 'react-router-dom'


function Nav() {
  return (
    <div className=" h-20 max-w-screen-lg mx-auto flex py-10 items-center justify-between border-b border-b-slate-100 ">
        <h1 className='text-xl font-bold tracking-tight'>ML Nexus</h1>
        <div className="flex gap-10  ">
            <Link to={'/'}>Home</Link>
            <Link to={'/'}>About</Link>
            <Link to={'/'}>Contact</Link>
        </div>
    </div>
)
}

export default Nav