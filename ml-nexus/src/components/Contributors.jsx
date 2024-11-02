import React, { useContext } from 'react'
import { repoContext } from '../utils/Context'

function Contributors() {
    const { contributors } = useContext(repoContext)
    return (
        <>
            <div className="p-10 ">
                <h1 className='text-4xl font-bold'>Our Contributors</h1>
                <div className="px-20 mt-6 flex flex-wrap gap-6">
                    {contributors.map((c, i) => <div key={i} className="md:h-48 w-48 bg-[#324655] rounded-lg flex flex-col items-center justify-between">
                        <div className="h-2/3 w-2/3 rounded-full overflow-hidden bg-gray-700 mt-2">
                        <img className='h-full w-full object-cover' src={c.avatar_url} alt={c.avatar_url} />
                        </div>
                        <h1 className='md:text-base text-xs mb-2 bg-teal-800 px-4 py-2 rounded-full'>{c.login}</h1>
                    </div>)}
                </div>
            </div>
        </>
    )
}

export default Contributors