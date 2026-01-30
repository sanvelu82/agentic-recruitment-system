import { useState, useEffect, useCallback, useRef } from 'react'
import PipelineStepper from './components/PipelineStepper'
import CandidateTable from './components/CandidateTable'
import HumanReviewModal from './components/HumanReviewModal'
import AuditTrailModal from './components/AuditTrailModal'
import BiasAuditPanel from './components/BiasAuditPanel'
import Sidebar from './components/Sidebar'
import TestingCenter from './components/TestingCenter'
import { ToastProvider, useToast } from './components/Toast'
import { mockPipelineData } from './data/mockData'
import { healthApi, jobsApi, candidatesApi, pipelineApi, reviewApi, testApi, PipelinePoller } from './services/api'
import { 
  Briefcase, 
  RefreshCw, 
  AlertCircle, 
  Plus, 
  Play, 
  Server, 
  ServerOff,
  Users,
  Trophy,
  Shield,
  Sparkles,
  X,
  Bell,
  ChevronRight,
  Loader2
} from 'lucide-react'

// Stats Card Component
function StatCard({ icon: Icon, label, value, color = 'indigo' }) {
  const colorClasses = {
    indigo: 'from-indigo-600 to-indigo-400 shadow-indigo-500/30',
    emerald: 'from-emerald-600 to-emerald-400 shadow-emerald-500/30',
    amber: 'from-amber-600 to-amber-400 shadow-amber-500/30',
    red: 'from-red-600 to-red-400 shadow-red-500/30',
  }

  return (
    <div className="glass-card p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-surface-400 mb-1">{label}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
        </div>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colorClasses[color]} shadow-lg flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  )
}

// Review Required Banner
function ReviewBanner({ onReviewClick, onDismiss }) {
  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 animate-bounce-in">
      <div className="glass-card border-amber-500/50 bg-amber-500/10 p-4 flex items-center gap-4 shadow-2xl shadow-amber-500/20">
        <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center">
          <Bell className="w-6 h-6 text-amber-400 animate-pulse" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-amber-400">Manual Review Required</h3>
          <p className="text-sm text-surface-300">The pipeline is awaiting human review before proceeding.</p>
        </div>
        <button onClick={onReviewClick} className="btn-primary ml-4">
          Review Now
          <ChevronRight className="w-4 h-4" />
        </button>
        <button onClick={onDismiss} className="p-2 hover:bg-white/5 rounded-lg transition-colors">
          <X className="w-5 h-5 text-surface-400" />
        </button>
      </div>
    </div>
  )
}

// Loading Skeleton
function DashboardSkeleton() {
  return (
    <div className="space-y-8 animate-pulse">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="glass-card p-5">
            <div className="h-4 w-24 skeleton rounded mb-2" />
            <div className="h-8 w-16 skeleton rounded" />
          </div>
        ))}
      </div>
      <div className="glass-card p-6">
        <div className="h-6 w-48 skeleton rounded mb-4" />
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 skeleton rounded" />
          ))}
        </div>
      </div>
    </div>
  )
}

// Main App Component
function AppContent() {
  const toast = useToast()
  const pollerRef = useRef(null)
  
  // State
  const [activeView, setActiveView] = useState('dashboard')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [pipelineState, setPipelineState] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showHumanReview, setShowHumanReview] = useState(false)
  const [reviewCandidate, setReviewCandidate] = useState(null)
  const [showAuditTrail, setShowAuditTrail] = useState(false)
  const [auditTrailCandidate, setAuditTrailCandidate] = useState(null)
  const [backendConnected, setBackendConnected] = useState(false)
  const [useMockData, setUseMockData] = useState(false)
  const [showReviewBanner, setShowReviewBanner] = useState(false)
  const [showTestingCenter, setShowTestingCenter] = useState(false)
  const [testQuestions, setTestQuestions] = useState([])
  const [testCandidate, setTestCandidate] = useState(null)
  
  // Pipeline management state
  const [jobs, setJobs] = useState([])
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [currentPipelineId, setCurrentPipelineId] = useState(null)
  const [showCreateJob, setShowCreateJob] = useState(false)
  const [newJob, setNewJob] = useState({ title: '', description: '', company: '' })
  const [newResume, setNewResume] = useState('')
  const [pipelineRunning, setPipelineRunning] = useState(false)

  // Check backend connection
  const checkBackendConnection = useCallback(async () => {
    try {
      const health = await healthApi.check()
      setBackendConnected(health.status === 'healthy')
      toast.success('Connected to backend API')
      return true
    } catch (err) {
      console.log('Backend not available, using mock data')
      setBackendConnected(false)
      toast.warning('Backend not available. Using demo mode.')
      return false
    }
  }, [toast])

  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    if (!backendConnected) return
    try {
      const response = await jobsApi.getJobs()
      setJobs(response.jobs || [])
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
      toast.error('Failed to fetch jobs')
    }
  }, [backendConnected, toast])

  // Fetch pipeline state
  const fetchPipelineState = useCallback(async () => {
    try {
      setLoading(true)
      
      if (useMockData || !backendConnected) {
        await new Promise(resolve => setTimeout(resolve, 500))
        setPipelineState(mockPipelineData)
        setError(null)
        return
      }

      if (currentPipelineId) {
        const pipeline = await pipelineApi.getPipeline(currentPipelineId)
        setPipelineState(pipeline.state || pipeline)
      } else {
        setPipelineState(null)
      }
      setError(null)
    } catch (err) {
      setError('Failed to fetch pipeline state')
      toast.error(err.message || 'Failed to fetch pipeline state')
    } finally {
      setLoading(false)
    }
  }, [backendConnected, useMockData, currentPipelineId, toast])

  // Setup polling
  useEffect(() => {
    if (currentPipelineId && backendConnected) {
      pollerRef.current = new PipelinePoller(currentPipelineId, {
        onStatusChange: (status, data) => {
          setPipelineState(data.state || data)
        },
        onAwaitingReview: () => {
          setShowReviewBanner(true)
          toast.warning('Manual review required', { title: 'Pipeline Paused' })
        },
        onCompleted: (data) => {
          setPipelineState(data.state || data)
          toast.success('Pipeline completed successfully!', { title: 'Success' })
          setActiveView('dashboard')
        },
        onFailed: (data) => {
          setPipelineState(data.state || data)
          toast.error(data.error_reason || 'Pipeline failed', { title: 'Error' })
        },
        onError: (err) => {
          toast.error(err.message)
        },
      })
      pollerRef.current.start()
    }

    return () => {
      if (pollerRef.current) {
        pollerRef.current.stop()
      }
    }
  }, [currentPipelineId, backendConnected, toast])

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

  // Handle review banner from pipeline state
  useEffect(() => {
    if (pipelineState?.current_stage === 'awaiting_human_review') {
      setShowReviewBanner(true)
    }
  }, [pipelineState])

  // Create job
  const handleCreateJob = async () => {
    if (!newJob.title || !newJob.description) {
      toast.warning('Please provide job title and description')
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
      toast.success('Job created successfully!')
    } catch (err) {
      toast.error(err.message || 'Failed to create job')
    }
  }

  // Add candidate
  const handleAddCandidate = async () => {
    if (!selectedJobId || !newResume) {
      toast.warning('Please select a job and provide resume text')
      return
    }
    try {
      await candidatesApi.addCandidate(selectedJobId, { resume_text: newResume })
      setNewResume('')
      toast.success('Candidate added successfully!')
    } catch (err) {
      toast.error(err.message || 'Failed to add candidate')
    }
  }

  // Run pipeline
  const handleRunPipeline = async () => {
    if (!selectedJobId) {
      toast.warning('Please select a job first')
      return
    }
    try {
      setPipelineRunning(true)
      toast.info('Starting pipeline...')
      
      const createResponse = await pipelineApi.createPipeline(selectedJobId)
      setCurrentPipelineId(createResponse.pipeline_id)
      
      const runResponse = await pipelineApi.runPipeline(createResponse.pipeline_id)
      await fetchPipelineState()
      
      toast.success(`Pipeline ${runResponse.status}!`)
    } catch (err) {
      toast.error(err.message || 'Failed to run pipeline')
    } finally {
      setPipelineRunning(false)
    }
  }

  // View audit trail
  const handleViewAuditTrail = (candidate) => {
    setAuditTrailCandidate(candidate)
    setShowAuditTrail(true)
  }

  // Request review
  const handleReviewRequest = (candidate) => {
    setReviewCandidate(candidate)
    setShowHumanReview(true)
    setShowReviewBanner(false)
  }

  // Submit review
  const handleReviewDecision = async (candidateId, decision, notes) => {
    try {
      if (backendConnected && currentPipelineId) {
        await reviewApi.submitReview(currentPipelineId, decision === 'approve', notes)
        toast.success(decision === 'approve' ? 'Candidate approved!' : 'Candidate rejected')
      }
      
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
      setShowReviewBanner(false)
    } catch (err) {
      toast.error(err.message || 'Failed to submit review')
    }
  }

  // Submit test
  const handleSubmitTest = async (testData) => {
    try {
      if (backendConnected && currentPipelineId) {
        await testApi.submitTest(currentPipelineId, testData.candidate_id, testData.responses)
        toast.success('Test submitted successfully!')
      }
      setShowTestingCenter(false)
      setTestCandidate(null)
    } catch (err) {
      toast.error(err.message || 'Failed to submit test')
    }
  }

  // Main content padding based on sidebar
  const mainPadding = sidebarCollapsed ? 'pl-20' : 'pl-64'

  return (
    <div className="min-h-screen">
      {/* Sidebar */}
      <Sidebar
        activeView={activeView}
        onViewChange={setActiveView}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        pipelineCount={currentPipelineId ? 1 : 0}
      />

      {/* Review Banner */}
      {showReviewBanner && (
        <ReviewBanner
          onReviewClick={() => {
            const candidateToReview = pipelineState?.final_rankings?.find(
              c => c.human_review_required && !c.human_review_completed
            )
            if (candidateToReview) handleReviewRequest(candidateToReview)
          }}
          onDismiss={() => setShowReviewBanner(false)}
        />
      )}

      {/* Main Content */}
      <main className={`${mainPadding} min-h-screen transition-all duration-300`}>
        {/* Header */}
        <header className="sticky top-0 z-30 bg-surface-950/80 backdrop-blur-xl border-b border-surface-800">
          <div className="px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-white">
                  {activeView === 'dashboard' && 'Dashboard'}
                  {activeView === 'jobs' && 'Job Management'}
                  {activeView === 'candidates' && 'Candidates'}
                  {activeView === 'pipelines' && 'Active Pipelines'}
                </h1>
                <p className="text-sm text-surface-400 mt-0.5">
                  {pipelineState?.job_id 
                    ? `Pipeline: ${currentPipelineId?.slice(0, 8)}...`
                    : 'No active pipeline'}
                </p>
              </div>
              
              <div className="flex items-center gap-4">
                {/* Connection Status */}
                <div className={`
                  flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
                  ${backendConnected 
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' 
                    : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                  }
                `}>
                  {backendConnected ? (
                    <>
                      <Server className="w-4 h-4" />
                      <span>API Connected</span>
                    </>
                  ) : (
                    <>
                      <ServerOff className="w-4 h-4" />
                      <span>Demo Mode</span>
                    </>
                  )}
                </div>
                
                <button 
                  onClick={fetchPipelineState}
                  className="btn-secondary"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="px-6 lg:px-8 py-8">
          {loading && !pipelineState ? (
            <DashboardSkeleton />
          ) : error ? (
            <div className="glass-card p-12 text-center">
              <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Something went wrong</h3>
              <p className="text-surface-400 mb-6">{error}</p>
              <button onClick={fetchPipelineState} className="btn-primary">
                Try Again
              </button>
            </div>
          ) : (
            <>
              {/* Pipeline Setup Panel */}
              {backendConnected && (
                <section className="mb-8">
                  <div className="glass-card p-6">
                    <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-indigo-400" />
                      Pipeline Setup
                    </h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      {/* Job Selection */}
                      <div>
                        <label className="block text-sm font-medium text-surface-300 mb-2">
                          Select Job
                        </label>
                        <select
                          value={selectedJobId || ''}
                          onChange={(e) => setSelectedJobId(e.target.value)}
                          className="select-field"
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
                          className="btn-secondary"
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
                          className="btn-primary disabled:opacity-50"
                        >
                          {pipelineRunning ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
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
                      <div className="border-t border-surface-800 pt-6 mt-6">
                        <h3 className="text-md font-semibold text-white mb-4">Create New Job</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <input
                            type="text"
                            placeholder="Job Title"
                            value={newJob.title}
                            onChange={(e) => setNewJob({ ...newJob, title: e.target.value })}
                            className="input-field"
                          />
                          <input
                            type="text"
                            placeholder="Company Name"
                            value={newJob.company}
                            onChange={(e) => setNewJob({ ...newJob, company: e.target.value })}
                            className="input-field"
                          />
                          <textarea
                            placeholder="Job Description (min 50 characters)"
                            value={newJob.description}
                            onChange={(e) => setNewJob({ ...newJob, description: e.target.value })}
                            className="textarea-field md:col-span-2 h-32"
                          />
                          <button onClick={handleCreateJob} className="btn-success">
                            Create Job
                          </button>
                        </div>
                      </div>
                    )}
                    
                    {/* Add Candidate Form */}
                    {selectedJobId && (
                      <div className="border-t border-surface-800 pt-6 mt-6">
                        <h3 className="text-md font-semibold text-white mb-4">Add Candidate Resume</h3>
                        <div className="flex gap-4">
                          <textarea
                            placeholder="Paste resume text here (min 100 characters)..."
                            value={newResume}
                            onChange={(e) => setNewResume(e.target.value)}
                            className="textarea-field flex-1 h-24"
                          />
                          <button onClick={handleAddCandidate} className="btn-secondary self-end">
                            Add Candidate
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Pipeline Progress */}
              {pipelineState && (
                <section className="mb-8">
                  <PipelineStepper 
                    currentStage={pipelineState?.current_stage} 
                    stages={pipelineState?.completed_stages || []}
                    errorReason={pipelineState?.error_reason}
                  />
                </section>
              )}

              {/* Stats Overview */}
              {pipelineState && (
                <section className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                  <StatCard
                    icon={Users}
                    label="Total Candidates"
                    value={pipelineState?.candidates?.length || 0}
                    color="indigo"
                  />
                  <StatCard
                    icon={Trophy}
                    label="Shortlisted"
                    value={pipelineState?.shortlisted_candidates?.length || 0}
                    color="emerald"
                  />
                  <StatCard
                    icon={AlertCircle}
                    label="Pending Review"
                    value={pipelineState?.final_rankings?.filter(r => r.human_review_required && !r.human_review_completed).length || 0}
                    color="amber"
                  />
                  <StatCard
                    icon={Shield}
                    label="Fairness Score"
                    value={pipelineState?.bias_audit_results?.overall_fairness_score 
                      ? `${(pipelineState.bias_audit_results.overall_fairness_score * 100).toFixed(0)}%`
                      : 'N/A'}
                    color={pipelineState?.bias_audit_results?.overall_fairness_score >= 0.8 ? 'emerald' : 'amber'}
                  />
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
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                      <Trophy className="w-5 h-5 text-amber-400" />
                      Candidate Rankings
                    </h2>
                    <span className="text-sm text-surface-400">
                      {pipelineState?.final_rankings?.length || 0} candidates
                    </span>
                  </div>
                  <CandidateTable 
                    candidates={pipelineState?.final_rankings || []}
                    onViewAuditTrail={handleViewAuditTrail}
                    onReviewRequest={handleReviewRequest}
                  />
                </section>
              )}
              
              {/* Empty State */}
              {!pipelineState && !loading && (
                <div className="glass-card p-16 text-center">
                  <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-indigo-600 to-indigo-400 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-indigo-500/30">
                    <Briefcase className="w-12 h-12 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-3">No Active Pipeline</h3>
                  <p className="text-surface-400 max-w-md mx-auto mb-8">
                    {backendConnected 
                      ? 'Create a job and add candidates to start a recruitment pipeline.'
                      : 'Start the backend server to connect, or continue with demo data.'}
                  </p>
                  {!backendConnected && (
                    <button 
                      onClick={() => { setUseMockData(true); fetchPipelineState(); }}
                      className="btn-primary"
                    >
                      <Sparkles className="w-5 h-5" />
                      Load Demo Data
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </main>

      {/* Testing Center Modal */}
      {showTestingCenter && (
        <TestingCenter
          questions={testQuestions}
          candidateId={testCandidate?.candidate_id}
          pipelineId={currentPipelineId}
          onSubmit={handleSubmitTest}
          onClose={() => {
            setShowTestingCenter(false)
            setTestCandidate(null)
          }}
        />
      )}

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

// App with Toast Provider
function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  )
}

export default App
