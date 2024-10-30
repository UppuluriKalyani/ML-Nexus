import React from 'react'

function Contributors() {
  return (
   <>
    <div className="p-10 ">
        <h1 className='text-4xl'>Our Contributors</h1>
        <div className="px-20 mt-4 flex flex-wrap gap-5">
            <div className="h-56 w-56 bg-[#324655] rounded-lg flex flex-col items-center justify-between">
                <div className="h-2/3 w-2/3 rounded-full bg-gray-700 mt-2"></div>
                <h1 className='text-xl mb-2 bg-teal-800 px-4 py-2 rounded-full'>Name</h1>
            </div>
        </div>
    </div>
   </>
  )
}

export default Contributors