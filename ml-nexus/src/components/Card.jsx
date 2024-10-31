import React from 'react'

function Card({data}) {
  return (
    

<div className="w-60 h-72 bg-white border border-gray-200 rounded-lg shadow dark:bg-[#324655] dark:border-gray-700">
    <a href="#">
        <img className="rounded-t-lg h-36 w-full object-cover" src={data.img} alt="" />
    </a>
    <div className="px-5 py-3">
        <a href="#">
            <h5 className="mb-2 text-xl font-bold tracking-tight text-gray-900 dark:text-white">{data.title}</h5>
        </a>
        <p className="mb-3 font-normal text-sm text-gray-700 dark:text-gray-400">{data.desc.split(" ").splice(0,8).join(" ")}
        </p>
    </div>
</div>

  )
}

export default Card