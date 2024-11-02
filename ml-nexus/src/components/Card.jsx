import React from 'react'

function Card({data}) {
  return (
    

<div className="lg:w-60 lg:h-72 group hover:dark:bg-[#3A5064] hover:text-[#B9D1DA] bg-white border border-gray-200 rounded-lg shadow dark:bg-[#324655] dark:border-gray-700 overflow-hidden">
    <a href="#">
        <img className="rounded-t-lg h-36 w-full object-cover group-hover:scale-105 ease-in duration-200" src={data.img} alt="" />
    </a>
    <div className="px-5 py-3">
        <a href="#">
            <h5 className="mb-2 text-xl font-bold tracking-tight text-gray-900 dark:text-white">{data.title}</h5>
        </a>
        <p className="mb-3 font-normal text-sm text-gray-700 dark:text-gray-400">{data.desc.split(" ").splice(0,8).join(" ") + "..."} 
          <a className='text-[#61B3A0] group-hover:opacity-100 ease-in-out duration-200 opacity-0' href={data.ref}>Explore more</a>
        </p>
    </div>
</div>

  )
}

export default Card