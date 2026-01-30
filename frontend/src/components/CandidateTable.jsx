import { useState } from 'react'
import { 
  Eye, 
  AlertTriangle, 
  CheckCircle, 
  ChevronUp, 
  ChevronDown,
  Star,
  TrendingUp,
  TrendingDown,
  Minus,
  Award,
  Sparkles,
  User,
  FileText
} from 'lucide-react'

// Loading Skeleton Component
function CandidateSkeleton() {
  return (
    <div className="glass-card p-6 animate-pulse">
      <div className="flex items-start gap-4">
        <div className="w-12 h-12 rounded-xl skeleton" />
        <div className="flex-1 space-y-3">
          <div className="h-4 w-32 skeleton rounded" />
          <div className="h-3 w-48 skeleton rounded" />
        </div>
        <div className="h-8 w-24 skeleton rounded-full" />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="h-16 skeleton rounded-xl" />
        <div className="h-16 skeleton rounded-xl" />
        <div className="h-16 skeleton rounded-xl" />
      </div>
    </div>
  )
}

// Empty State Component
function EmptyState() {
  return (
    <div className="glass-card p-12 text-center">
      <div className="w-20 h-20 rounded-2xl bg-surface-800 flex items-center justify-center mx-auto mb-6">
        <User className="w-10 h-10 text-surface-600" />
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">No Candidates Yet</h3>
      <p className="text-surface-400 max-w-md mx-auto">
        Candidates will appear here once they've been processed through the recruitment pipeline.
      </p>
    </div>
  )
}

function CandidateTable({ candidates, onViewAuditTrail, onReviewRequest, isLoading = false }) {
  const [sortField, setSortField] = useState('rank')
  const [sortDirection, setSortDirection] = useState('asc')
  const [expandedRow, setExpandedRow] = useState(null)

  // Sort candidates
  const sortedCandidates = [...candidates].sort((a, b) => {
    let aVal = a[sortField]
    let bVal = b[sortField]
    
    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase()
      bVal = bVal.toLowerCase()
    }
    
    if (sortDirection === 'asc') {
      return aVal > bVal ? 1 : -1
    }
    return aVal < bVal ? 1 : -1
  })

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const getScoreColor = (score) => {
    if (score >= 0.85) return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30'
    if (score >= 0.7) return 'text-indigo-400 bg-indigo-500/20 border-indigo-500/30'
    if (score >= 0.5) return 'text-amber-400 bg-amber-500/20 border-amber-500/30'
    return 'text-red-400 bg-red-500/20 border-red-500/30'
  }

  const getRecommendationBadge = (recommendation, score) => {
    // Check for strongly recommended (score > 0.8)
    if (score > 0.8 || recommendation === 'strongly_recommend') {
      return (
        <span className="badge-recommended">
          <Sparkles className="w-3.5 h-3.5" />
          Strongly Recommended
        </span>
      )
    }
    
    const badges = {
      recommend: { 
        className: 'badge bg-emerald-500/20 text-emerald-400 border-emerald-500/30', 
        label: 'Recommended',
        icon: CheckCircle 
      },
      consider: { 
        className: 'badge bg-amber-500/20 text-amber-400 border-amber-500/30', 
        label: 'Consider',
        icon: null 
      },
      not_recommended: { 
        className: 'badge bg-red-500/20 text-red-400 border-red-500/30', 
        label: 'Not Recommended',
        icon: null 
      },
    }
    
    const badge = badges[recommendation] || { 
      className: 'badge bg-surface-700 text-surface-300 border-surface-600', 
      label: recommendation?.replace(/_/g, ' ') || 'Pending'
    }
    
    return (
      <span className={badge.className}>
        {badge.icon && <badge.icon className="w-3 h-3" />}
        {badge.label}
      </span>
    )
  }

  const getRankIcon = (rank) => {
    if (rank === 1) return <Award className="w-5 h-5 text-yellow-400" />
    if (rank === 2) return <Award className="w-5 h-5 text-surface-300" />
    if (rank === 3) return <Award className="w-5 h-5 text-amber-600" />
    return null
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <CandidateSkeleton />
        <CandidateSkeleton />
        <CandidateSkeleton />
      </div>
    )
  }

  if (candidates.length === 0) {
    return <EmptyState />
  }

  return (
    <div className="space-y-4">
      {/* Sort Controls */}
      <div className="flex items-center gap-4 mb-2">
        <span className="text-sm text-surface-500">Sort by:</span>
        <div className="flex items-center gap-2">
          {[
            { field: 'rank', label: 'Rank' },
            { field: 'final_composite_score', label: 'Score' },
            { field: 'recommendation', label: 'Status' },
          ].map(({ field, label }) => (
            <button
              key={field}
              onClick={() => handleSort(field)}
              className={`
                px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                ${sortField === field 
                  ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30' 
                  : 'bg-surface-800 text-surface-400 hover:text-white hover:bg-surface-700'
                }
              `}
            >
              {label}
              {sortField === field && (
                sortDirection === 'asc' 
                  ? <ChevronUp className="w-4 h-4 inline ml-1" /> 
                  : <ChevronDown className="w-4 h-4 inline ml-1" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Candidate Cards */}
      {sortedCandidates.map((candidate) => (
        <div key={candidate.candidate_id} className="glass-card-hover">
          {/* Main Card Content */}
          <div className="p-6">
            <div className="flex items-start gap-4">
              {/* Rank Badge */}
              <div className={`
                flex-shrink-0 w-14 h-14 rounded-xl
                flex flex-col items-center justify-center
                ${candidate.rank <= 3 
                  ? 'bg-gradient-to-br from-indigo-600 to-indigo-400' 
                  : 'bg-surface-800'
                }
              `}>
                {getRankIcon(candidate.rank) || (
                  <span className="text-xl font-bold text-white">#{candidate.rank}</span>
                )}
                {candidate.rank <= 3 && (
                  <span className="text-xs text-white/80">#{candidate.rank}</span>
                )}
              </div>

              {/* Candidate Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="text-lg font-semibold text-white truncate">
                    Candidate {candidate.candidate_id.slice(0, 8)}
                  </h3>
                  {getRecommendationBadge(candidate.recommendation, candidate.final_composite_score)}
                </div>
                <p className="text-sm text-surface-400">
                  ID: {candidate.candidate_id}
                </p>
              </div>

              {/* Main Score */}
              <div className={`
                px-4 py-2 rounded-xl border text-center
                ${getScoreColor(candidate.final_composite_score)}
              `}>
                <div className="text-2xl font-bold">
                  {(candidate.final_composite_score * 100).toFixed(0)}%
                </div>
                <div className="text-xs opacity-80">Match Score</div>
              </div>
            </div>

            {/* Score Breakdown */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              {/* Resume Score */}
              <div className="glass-panel p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-surface-500">Resume</span>
                  <FileText className="w-4 h-4 text-surface-500" />
                </div>
                <div className="text-lg font-semibold text-white">
                  {(candidate.resume_match_score * 100).toFixed(0)}%
                </div>
              </div>

              {/* Test Score */}
              <div className="glass-panel p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-surface-500">Test</span>
                  {candidate.test_score > candidate.resume_match_score ? (
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                  ) : candidate.test_score < candidate.resume_match_score ? (
                    <TrendingDown className="w-4 h-4 text-red-400" />
                  ) : (
                    <Minus className="w-4 h-4 text-surface-500" />
                  )}
                </div>
                <div className="text-lg font-semibold text-white">
                  {(candidate.test_score * 100).toFixed(0)}%
                </div>
              </div>

              {/* Status */}
              <div className="glass-panel p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-surface-500">Status</span>
                </div>
                {candidate.human_review_required && !candidate.human_review_completed ? (
                  <span className="text-sm font-medium text-amber-400 flex items-center gap-1">
                    <AlertTriangle className="w-4 h-4" />
                    Review Needed
                  </span>
                ) : candidate.bias_audit_passed ? (
                  <span className="text-sm font-medium text-emerald-400 flex items-center gap-1">
                    <CheckCircle className="w-4 h-4" />
                    Audit Passed
                  </span>
                ) : (
                  <span className="text-sm font-medium text-surface-400">Pending</span>
                )}
              </div>

              {/* Actions */}
              <div className="glass-panel p-3 flex items-center justify-center gap-2">
                <button
                  onClick={() => onViewAuditTrail(candidate)}
                  className="btn-ghost text-xs px-3 py-1.5"
                >
                  <Eye className="w-4 h-4" />
                  Audit
                </button>
                {candidate.human_review_required && !candidate.human_review_completed && (
                  <button
                    onClick={() => onReviewRequest(candidate)}
                    className="btn-primary text-xs px-3 py-1.5"
                  >
                    Review
                  </button>
                )}
              </div>
            </div>

            {/* Expand Toggle */}
            <button
              onClick={() => setExpandedRow(
                expandedRow === candidate.candidate_id ? null : candidate.candidate_id
              )}
              className="w-full mt-4 flex items-center justify-center gap-2 py-2 text-sm text-surface-400 hover:text-white transition-colors"
            >
              {expandedRow === candidate.candidate_id ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Hide Details
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show Details
                </>
              )}
            </button>
          </div>

          {/* Expanded Details */}
          {expandedRow === candidate.candidate_id && (
            <div className="px-6 pb-6 border-t border-surface-800">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6">
                {/* Key Strengths */}
                <div className="glass-panel p-4">
                  <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    Key Strengths
                  </h4>
                  {candidate.key_strengths?.length > 0 ? (
                    <ul className="space-y-2">
                      {candidate.key_strengths.map((strength, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-surface-300">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 flex-shrink-0" />
                          {strength}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-surface-500 italic">No strengths recorded</p>
                  )}
                </div>

                {/* Key Concerns */}
                <div className="glass-panel p-4">
                  <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    Key Concerns
                  </h4>
                  {candidate.key_concerns?.length > 0 ? (
                    <ul className="space-y-2">
                      {candidate.key_concerns.map((concern, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-surface-300">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-2 flex-shrink-0" />
                          {concern}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-surface-500 italic">No concerns recorded</p>
                  )}
                </div>

                {/* Ranking Explanation */}
                <div className="md:col-span-2 glass-panel p-4">
                  <h4 className="text-sm font-semibold text-white mb-3">Ranking Explanation</h4>
                  <p className="text-sm text-surface-300 leading-relaxed">
                    {candidate.ranking_explanation || 'No explanation provided'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export default CandidateTable
