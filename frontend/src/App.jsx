import { useState, useEffect, useCallback } from 'react'
import PipelineStepper from './components/PipelineStepper'
import CandidateTable from './components/CandidateTable'
import HumanReviewModal from './components/HumanReviewModal'
import AuditTrailModal from './components/AuditTrailModal'
import BiasAuditPanel from './components/BiasAuditPanel'
import ResumeUploader from './components/ResumeUploader'
import { mockPipelineData } from './data/mockData'
import { healthApi, jobsApi, candidatesApi, pipelineApi, reviewApi } from './services/api'
import { Briefcase, RefreshCw, AlertCircle, Plus, Play, Server, ServerOff, FileText, Type } from 'lucide-react'

function App() {
  const [pipelineState, setPipelineState] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showHumanReview, setShowHumanReview] = useState(false)
  const [reviewCandidate, setReviewCandidate] = useState(null)
  const [showAuditTrail, setShowAuditTrail] = useState(false)
  const [auditTrailCandidate, setAuditTrailCandidate] = useState(null)
  const [backendConnected, setBackendConnected] = useState(false)
  const [useMockData, setUseMockData] = useState(false)
  
  // Pipeline management state
  const [jobs, setJobs] = useState([])
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [currentPipelineId, setCurrentPipelineId] = useState(null)
  const [showCreateJob, setShowCreateJob] = useState(false)
  const [newJob, setNewJob] = useState({ title: '', description: '', company: '' })
  const [newResume, setNewResume] = useState('')
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [resumeInputMode, setResumeInputMode] = useState('pdf') // 'pdf' or 'text'

  // Check backend connection
  const checkBackendConnection = useCallback(async () => {
    try {
      const health = await healthApi.check()
      setBackendConnected(health.status === 'healthy')
      return true
    } catch (err) {
      console.log('Backend not available, using mock data')
      setBackendConnected(false)
      return false
    }
  }, [])

  // Fetch jobs from backend
  const fetchJobs = useCallback(async () => {
    if (!backendConnected) return
    try {
      const response = await jobsApi.getJobs()
      setJobs(response.jobs || [])
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
    }
  }, [backendConnected])

  // Fetch pipeline state
  const fetchPipelineState = useCallback(async () => {
    try {
      setLoading(true)
      
      if (useMockData || !backendConnected) {
        // Use mock data
        await new Promise(resolve => setTimeout(resolve, 500))
        setPipelineState(mockPipelineData)
        setError(null)
        return
      }

      if (currentPipelineId) {
        // Fetch real pipeline data
        const pipeline = await pipelineApi.getPipeline(currentPipelineId)
        setPipelineState(pipeline.state || pipeline)
      } else {
        // No active pipeline
        setPipelineState(null)
      }
      setError(null)
    } catch (err) {
      setError('Failed to fetch pipeline state')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [backendConnected, useMockData, currentPipelineId])

  // Initialize
  useEffect(() => {
    const init = async () => {
      const connected = await checkBackendConnection()
      if (connected) {
        await fetchJobs()
      } else {
        setUseMockData(true)
      }
      await fetchPipelineState()
    }
    init()
  }, [checkBackendConnection, fetchJobs, fetchPipelineState])

  // Check if human review is needed
  useEffect(() => {
    if (pipelineState?.current_stage === 'awaiting_human_review') {
      setShowHumanReview(true)
    }
  }, [pipelineState])

  // Create a new job
  const handleCreateJob = async () => {
    if (!newJob.title || !newJob.description) {
      alert('Please provide job title and description')
      return
    }
    try {
      const response = await jobsApi.createJob({
        title: newJob.title,
        raw_description: newJob.description,
        company: newJob.company || 'My Company',
      })
      setSelectedJobId(response.job_id)
      setNewJob({ title: '', description: '', company: '' })
      setShowCreateJob(false)
      await fetchJobs()
    } catch (err) {
      console.error('Failed to create job:', err)
      alert('Failed to create job: ' + (err.response?.data?.detail || err.message))
    }
  }

  // Add a candidate to the selected job
  const handleAddCandidate = async () => {
    if (!selectedJobId || !newResume) {
      alert('Please select a job and provide resume text')
      return
    }
    try {
      await candidatesApi.addCandidate(selectedJobId, {
        resume_text: newResume,
      })
      setNewResume('')
      alert('Candidate added successfully!')
    } catch (err) {
      console.error('Failed to add candidate:', err)
      alert('Failed to add candidate: ' + (err.response?.data?.detail || err.message))
    }
  }

  // Create and run pipeline
  const handleRunPipeline = async () => {
    if (!selectedJobId) {
      alert('Please select a job first')
      return
    }
    try {
      setPipelineRunning(true)
      
      // Create pipeline
      const createResponse = await pipelineApi.createPipeline(selectedJobId)
      setCurrentPipelineId(createResponse.pipeline_id)
      
      // Run pipeline
      const runResponse = await pipelineApi.runPipeline(createResponse.pipeline_id)
      
      // Fetch updated state
      await fetchPipelineState()
      
      alert(`Pipeline completed! Status: ${runResponse.status}`)
    } catch (err) {
      console.error('Failed to run pipeline:', err)
      alert('Failed to run pipeline: ' + (err.response?.data?.detail || err.message))
    } finally {
      setPipelineRunning(false)
    }
  }

  // Handle viewing audit trail
  const handleViewAuditTrail = (candidate) => {
    setAuditTrailCandidate(candidate)
    setShowAuditTrail(true)
  }

  // Handle human review request
  const handleReviewRequest = (candidate) => {
    setReviewCandidate(candidate)
    setShowHumanReview(true)
  }

  // Handle review decision
  const handleReviewDecision = async (candidateId, decision, notes) => {
    try {
      if (backendConnected && currentPipelineId) {
        await reviewApi.submitReview(currentPipelineId, decision === 'approve', notes)
      }
      
      // Update local state
      setPipelineState(prev => ({
        ...prev,
        current_stage: decision === 'approve' ? 'completed' : prev.current_stage,
        final_rankings: prev.final_rankings?.map(r => 
          r.candidate_id === candidateId 
            ? { ...r, human_review_completed: true, human_decision: decision }
            : r
        )
      }))
      
      setShowHumanReview(false)
      setReviewCandidate(null)
    } catch (err) {
      console.error('Failed to submit review:', err)
      alert('Failed to submit review: ' + err.message)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className="w-8 h-8 text-primary-600 animate-spin" />
          <p className="text-gray-600">Loading pipeline data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-red-600">
          <AlertCircle className="w-12 h-12" />
          <p className="text-lg font-medium">{error}</p>
          <button onClick={fetchPipelineState} className="btn-primary">
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-100 rounded-lg">
                <Briefcase className="w-6 h-6 text-primary-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Recruitment Pipeline</h1>
                <p className="text-sm text-gray-500">
                  {pipelineState?.job_id 
                    ? `Job ID: ${pipelineState.job_id.slice(0, 8)}...`
                    : 'No active pipeline'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {/* Backend connection indicator */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                backendConnected 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-amber-100 text-amber-700'
              }`}>
                {backendConnected ? (
                  <>
                    <Server className="w-4 h-4" />
                    <span>API Connected</span>
                  </>
                ) : (
                  <>
                    <ServerOff className="w-4 h-4" />
                    <span>Mock Mode</span>
                  </>
                )}
              </div>
              <button 
                onClick={fetchPipelineState}
                className="btn-secondary flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Job Management Panel (when backend connected) */}
        {backendConnected && (
          <section className="mb-8">
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Setup</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {/* Job Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Select Job
                  </label>
                  <select
                    value={selectedJobId || ''}
                    onChange={(e) => setSelectedJobId(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="">-- Select a job --</option>
                    {jobs.map((job) => (
                      <option key={job.job_id} value={job.job_id}>
                        {job.title} ({job.company})
                      </option>
                    ))}
                  </select>
                </div>
                
                {/* Create Job Button */}
                <div className="flex items-end">
                  <button
                    onClick={() => setShowCreateJob(!showCreateJob)}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Plus className="w-4 h-4" />
                    New Job
                  </button>
                </div>
                
                {/* Run Pipeline Button */}
                <div className="flex items-end">
                  <button
                    onClick={handleRunPipeline}
                    disabled={!selectedJobId || pipelineRunning}
                    className="btn-primary flex items-center gap-2 disabled:opacity-50"
                  >
                    {pipelineRunning ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Run Pipeline
                      </>
                    )}
                  </button>
                </div>
              </div>
              
              {/* Create Job Form */}
              {showCreateJob && (
                <div className="border-t pt-4 mt-4">
                  <h3 className="text-md font-medium text-gray-800 mb-3">Create New Job</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <input
                      type="text"
                      placeholder="Job Title"
                      value={newJob.title}
                      onChange={(e) => setNewJob({ ...newJob, title: e.target.value })}
                      className="px-3 py-2 border border-gray-300 rounded-lg"
                    />
                    <input
                      type="text"
                      placeholder="Company Name"
                      value={newJob.company}
                      onChange={(e) => setNewJob({ ...newJob, company: e.target.value })}
                      className="px-3 py-2 border border-gray-300 rounded-lg"
                    />
                    <textarea
                      placeholder="Job Description (min 50 characters)"
                      value={newJob.description}
                      onChange={(e) => setNewJob({ ...newJob, description: e.target.value })}
                      className="px-3 py-2 border border-gray-300 rounded-lg md:col-span-2 h-32"
                    />
                    <button onClick={handleCreateJob} className="btn-primary">
                      Create Job
                    </button>
                  </div>
                </div>
              )}
              
              {/* Add Candidate Form */}
              {selectedJobId && (
                <div className="border-t pt-4 mt-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-md font-medium text-gray-800">Add Candidate Resume</h3>
                    
                    {/* Toggle between PDF and Text input */}
                    <div className="flex items-center bg-gray-100 rounded-lg p-1">
                      <button
                        onClick={() => setResumeInputMode('pdf')}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                          resumeInputMode === 'pdf'
                            ? 'bg-white text-primary-600 shadow-sm'
                            : 'text-gray-600 hover:text-gray-800'
                        }`}
                      >
                        <FileText className="w-4 h-4" />
                        PDF Upload
                      </button>
                      <button
                        onClick={() => setResumeInputMode('text')}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                          resumeInputMode === 'text'
                            ? 'bg-white text-primary-600 shadow-sm'
                            : 'text-gray-600 hover:text-gray-800'
                        }`}
                      >
                        <Type className="w-4 h-4" />
                        Paste Text
                      </button>
                    </div>
                  </div>
                  
                  {resumeInputMode === 'pdf' ? (
                    <ResumeUploader
                      jobId={selectedJobId}
                      onUploadSuccess={(result) => {
                        console.log('Resume uploaded:', result)
                      }}
                      onUploadError={(error) => {
                        console.error('Upload error:', error)
                      }}
                      disabled={pipelineRunning}
                    />
                  ) : (
                    <div className="flex gap-4">
                      <textarea
                        placeholder="Paste resume text here (min 100 characters)..."
                        value={newResume}
                        onChange={(e) => setNewResume(e.target.value)}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-lg h-24"
                      />
                      <button onClick={handleAddCandidate} className="btn-secondary self-end">
                        Add Candidate
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        {/* Pipeline Progress */}
        {pipelineState && (
          <section className="mb-8">
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-6">Pipeline Progress</h2>
              <PipelineStepper 
                currentStage={pipelineState?.current_stage} 
                stages={pipelineState?.completed_stages || []}
              />
            </div>
          </section>
        )}

        {/* Stats Overview */}
        {pipelineState && (
          <section className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="card">
              <p className="text-sm text-gray-500">Total Candidates</p>
              <p className="text-2xl font-bold text-gray-900">
                {pipelineState?.candidates?.length || 0}
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-500">Shortlisted</p>
              <p className="text-2xl font-bold text-green-600">
                {pipelineState?.shortlisted_candidates?.length || 0}
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-500">Pending Review</p>
              <p className="text-2xl font-bold text-amber-600">
                {pipelineState?.final_rankings?.filter(r => r.human_review_required && !r.human_review_completed).length || 0}
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-500">Fairness Score</p>
              <p className="text-2xl font-bold text-primary-600">
                {pipelineState?.bias_audit_results?.overall_fairness_score 
                  ? `${(pipelineState.bias_audit_results.overall_fairness_score * 100).toFixed(0)}%`
                  : 'N/A'}
              </p>
            </div>
          </section>
        )}

        {/* Bias Audit Panel */}
        {pipelineState?.bias_audit_results && (
          <section className="mb-8">
            <BiasAuditPanel auditResults={pipelineState.bias_audit_results} />
          </section>
        )}

        {/* Candidate Rankings */}
        {pipelineState?.final_rankings && (
          <section>
            <div className="card">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">Candidate Rankings</h2>
                <span className="text-sm text-gray-500">
                  Showing {pipelineState?.final_rankings?.length || 0} candidates
                </span>
              </div>
              <CandidateTable 
                candidates={pipelineState?.final_rankings || []}
                onViewAuditTrail={handleViewAuditTrail}
                onReviewRequest={handleReviewRequest}
              />
            </div>
          </section>
        )}
        
        {/* Empty state */}
        {!pipelineState && !loading && (
          <div className="text-center py-16">
            <Briefcase className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Active Pipeline</h3>
            <p className="text-gray-500 mb-4">
              {backendConnected 
                ? 'Create a job and add candidates to start a recruitment pipeline.'
                : 'Start the backend server to connect, or continue with mock data.'}
            </p>
            {!backendConnected && (
              <button 
                onClick={() => { setUseMockData(true); fetchPipelineState(); }}
                className="btn-primary"
              >
                Load Demo Data
              </button>
            )}
          </div>
        )}
      </main>

      {/* Human Review Modal */}
      <HumanReviewModal
        isOpen={showHumanReview}
        onClose={() => {
          setShowHumanReview(false)
          setReviewCandidate(null)
        }}
        candidate={reviewCandidate}
        biasFindings={pipelineState?.bias_audit_results?.findings || []}
        onDecision={handleReviewDecision}
      />

      {/* Audit Trail Modal */}
      <AuditTrailModal
        isOpen={showAuditTrail}
        onClose={() => {
          setShowAuditTrail(false)
          setAuditTrailCandidate(null)
        }}
        candidate={auditTrailCandidate}
      />
    </div>
  )
}

export default App
