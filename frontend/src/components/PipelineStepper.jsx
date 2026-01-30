import { 
  FileText, 
  Users, 
  GitCompare, 
  Filter, 
  FileQuestion, 
  ClipboardCheck, 
  Trophy, 
  Shield, 
  CheckCircle2,
  Clock,
  AlertCircle,
  XCircle,
  Loader2,
  Brain,
  Sparkles
} from 'lucide-react'

// Pipeline stages configuration with agent names
const PIPELINE_STAGES = [
  { 
    id: 'initialized', 
    label: 'Pipeline Initialized', 
    agent: 'System',
    icon: Clock,
    description: 'Setting up recruitment pipeline'
  },
  { 
    id: 'jd_analysis', 
    label: 'Analyzing Job Description', 
    agent: 'JD Analyzer',
    icon: FileText,
    description: 'Extracting requirements, skills, and qualifications'
  },
  { 
    id: 'resume_parsing', 
    label: 'Parsing Resumes', 
    agent: 'Resume Parser',
    icon: Users,
    description: 'Extracting candidate information and skills'
  },
  { 
    id: 'matching', 
    label: 'Matching Candidates', 
    agent: 'Matcher Agent',
    icon: GitCompare,
    description: 'Calculating match scores against job requirements'
  },
  { 
    id: 'shortlisting', 
    label: 'Shortlisting', 
    agent: 'Shortlister',
    icon: Filter,
    description: 'Filtering top candidates based on match scores'
  },
  { 
    id: 'test_generation', 
    label: 'Generating Tests', 
    agent: 'Test Generator',
    icon: FileQuestion,
    description: 'Creating technical assessment questions'
  },
  { 
    id: 'test_evaluation', 
    label: 'Evaluating Tests', 
    agent: 'Test Evaluator',
    icon: ClipboardCheck,
    description: 'Scoring candidate responses'
  },
  { 
    id: 'ranking', 
    label: 'Ranking Candidates', 
    agent: 'Ranker Agent',
    icon: Trophy,
    description: 'Generating final candidate rankings'
  },
  { 
    id: 'bias_audit', 
    label: 'Auditing for Bias', 
    agent: 'Bias Auditor',
    icon: Shield,
    description: 'Ensuring fairness and compliance'
  },
]

// Terminal states
const TERMINAL_STATES = {
  completed: { 
    label: 'Pipeline Completed', 
    color: 'emerald', 
    icon: CheckCircle2,
    description: 'All stages completed successfully'
  },
  failed: { 
    label: 'Pipeline Failed', 
    color: 'red', 
    icon: XCircle,
    description: 'An error occurred during processing'
  },
  awaiting_human_review: { 
    label: 'Awaiting Human Review', 
    color: 'amber', 
    icon: AlertCircle,
    description: 'Manual review required before proceeding'
  },
}

function PipelineStepper({ currentStage, stages = [], errorReason = null }) {
  const currentIndex = PIPELINE_STAGES.findIndex(s => s.id === currentStage)
  const isTerminalState = currentStage in TERMINAL_STATES
  const terminalInfo = TERMINAL_STATES[currentStage]

  const getStageStatus = (stage, index) => {
    if (stages.includes(stage.id)) return 'completed'
    if (stage.id === currentStage && !isTerminalState) return 'current'
    if (index < currentIndex) return 'completed'
    return 'pending'
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-400 flex items-center justify-center">
          <Brain className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">Live Agent Activity</h3>
          <p className="text-sm text-surface-400">Real-time pipeline execution log</p>
        </div>
      </div>

      {/* Terminal State Banner */}
      {isTerminalState && (
        <div className={`
          mb-6 p-4 rounded-xl border
          ${currentStage === 'completed' 
            ? 'bg-emerald-500/10 border-emerald-500/30' 
            : currentStage === 'failed'
            ? 'bg-red-500/10 border-red-500/30'
            : 'bg-amber-500/10 border-amber-500/30'
          }
        `}>
          <div className="flex items-center gap-3">
            {terminalInfo && (
              <>
                <terminalInfo.icon className={`w-6 h-6 ${
                  currentStage === 'completed' ? 'text-emerald-400' :
                  currentStage === 'failed' ? 'text-red-400' : 'text-amber-400'
                }`} />
                <div>
                  <p className={`font-semibold ${
                    currentStage === 'completed' ? 'text-emerald-400' :
                    currentStage === 'failed' ? 'text-red-400' : 'text-amber-400'
                  }`}>
                    {terminalInfo.label}
                  </p>
                  <p className="text-sm text-surface-400">{terminalInfo.description}</p>
                </div>
              </>
            )}
          </div>
          
          {/* Error Reason */}
          {currentStage === 'failed' && errorReason && (
            <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <p className="text-sm text-red-300">
                <span className="font-semibold">Error: </span>
                {errorReason}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Agent Log */}
      <div className="space-y-1">
        {PIPELINE_STAGES.map((stage, index) => {
          const status = getStageStatus(stage, index)
          const Icon = stage.icon
          const isActive = status === 'current'
          const isCompleted = status === 'completed'
          const isPending = status === 'pending'

          return (
            <div key={stage.id} className="relative">
              {/* Connector Line */}
              {index < PIPELINE_STAGES.length - 1 && (
                <div 
                  className={`
                    absolute left-5 top-12 w-0.5 h-8
                    ${isCompleted ? 'bg-emerald-500' : 'bg-surface-700'}
                  `}
                />
              )}

              <div 
                className={`
                  flex items-start gap-4 p-3 rounded-xl
                  transition-all duration-300
                  ${isActive ? 'bg-indigo-500/10 border border-indigo-500/30' : ''}
                  ${isCompleted ? 'opacity-100' : ''}
                  ${isPending ? 'opacity-50' : ''}
                `}
              >
                {/* Status Icon */}
                <div className={`
                  relative flex-shrink-0 w-10 h-10 rounded-xl
                  flex items-center justify-center
                  transition-all duration-300
                  ${isCompleted 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : isActive 
                    ? 'bg-indigo-500/20 text-indigo-400' 
                    : 'bg-surface-800 text-surface-500'
                  }
                `}>
                  {isActive ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : isCompleted ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : (
                    <Icon className="w-5 h-5" />
                  )}
                  
                  {/* Pulse ring for active */}
                  {isActive && (
                    <div className="absolute inset-0 rounded-xl bg-indigo-500/20 animate-ping" />
                  )}
                </div>

                {/* Stage Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`
                      text-sm font-semibold
                      ${isCompleted ? 'text-emerald-400' : isActive ? 'text-white' : 'text-surface-500'}
                    `}>
                      {stage.label}
                    </span>
                    
                    {isActive && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-500/20 text-indigo-400 animate-pulse">
                        <Sparkles className="w-3 h-3" />
                        Active
                      </span>
                    )}
                  </div>

                  <p className={`
                    text-xs
                    ${isActive ? 'text-surface-300' : 'text-surface-500'}
                  `}>
                    {isActive ? (
                      <span className="flex items-center gap-1">
                        <span className="font-medium text-indigo-400">{stage.agent}</span>
                        <span>is thinking...</span>
                      </span>
                    ) : isCompleted ? (
                      <span className="flex items-center gap-1">
                        <span className="text-emerald-400">{stage.agent}</span>
                        <span>completed</span>
                      </span>
                    ) : (
                      stage.description
                    )}
                  </p>
                </div>

                {/* Timestamp placeholder for completed */}
                {isCompleted && (
                  <span className="text-xs text-surface-500 flex-shrink-0">
                    ✓
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PipelineStepper
