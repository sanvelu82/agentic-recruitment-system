import { useState } from 'react'
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  ChevronDown, 
  ChevronUp,
  XCircle,
  AlertCircle,
  Info,
  Eye,
  Scale,
  TrendingUp
} from 'lucide-react'

// Circular Progress Component
function CircularProgress({ value, size = 120, strokeWidth = 8, label }) {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (value / 100) * circumference
  
  const getColor = (val) => {
    if (val >= 90) return { stroke: '#10B981', text: 'text-emerald-400', bg: 'text-emerald-500/20' }
    if (val >= 70) return { stroke: '#6366F1', text: 'text-indigo-400', bg: 'text-indigo-500/20' }
    if (val >= 50) return { stroke: '#F59E0B', text: 'text-amber-400', bg: 'text-amber-500/20' }
    return { stroke: '#EF4444', text: 'text-red-400', bg: 'text-red-500/20' }
  }
  
  const colors = getColor(value)

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-surface-700"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={colors.stroke}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{
            filter: `drop-shadow(0 0 6px ${colors.stroke}40)`
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-bold ${colors.text}`}>
          {value.toFixed(0)}%
        </span>
        {label && (
          <span className="text-xs text-surface-400 mt-1">{label}</span>
        )}
      </div>
    </div>
  )
}

function BiasAuditPanel({ auditResults }) {
  const [isExpanded, setIsExpanded] = useState(true)

  if (!auditResults) return null

  const { 
    audit_passed, 
    overall_fairness_score, 
    findings = [], 
    recommendations = [],
    compliance_notes = []
  } = auditResults

  const fairnessPercent = (overall_fairness_score || 0) * 100

  const getSeverityCount = (severity) => {
    return findings.filter(f => f.severity === severity).length
  }

  const getSeverityStyles = (severity) => {
    switch (severity) {
      case 'critical':
        return { 
          bg: 'bg-red-500/10', 
          text: 'text-red-400', 
          border: 'border-red-500/30', 
          icon: XCircle,
          badge: 'bg-red-500/20 text-red-400'
        }
      case 'high':
        return { 
          bg: 'bg-orange-500/10', 
          text: 'text-orange-400', 
          border: 'border-orange-500/30', 
          icon: AlertTriangle,
          badge: 'bg-orange-500/20 text-orange-400'
        }
      case 'medium':
        return { 
          bg: 'bg-amber-500/10', 
          text: 'text-amber-400', 
          border: 'border-amber-500/30', 
          icon: AlertCircle,
          badge: 'bg-amber-500/20 text-amber-400'
        }
      case 'low':
        return { 
          bg: 'bg-indigo-500/10', 
          text: 'text-indigo-400', 
          border: 'border-indigo-500/30', 
          icon: Info,
          badge: 'bg-indigo-500/20 text-indigo-400'
        }
      default:
        return { 
          bg: 'bg-surface-800', 
          text: 'text-surface-400', 
          border: 'border-surface-700', 
          icon: Info,
          badge: 'bg-surface-700 text-surface-400'
        }
    }
  }

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div 
        className="flex items-center justify-between p-6 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-4">
          <div className={`
            p-3 rounded-xl
            ${audit_passed 
              ? 'bg-emerald-500/20 text-emerald-400' 
              : 'bg-amber-500/20 text-amber-400'
            }
          `}>
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold text-white">Transparency Dashboard</h3>
              {audit_passed ? (
                <span className="badge-success">
                  <CheckCircle className="w-3 h-3" />
                  Audit Passed
                </span>
              ) : (
                <span className="badge-warning">
                  <AlertTriangle className="w-3 h-3" />
                  Review Required
                </span>
              )}
            </div>
            <p className="text-sm text-surface-400 mt-0.5">
              Fairness Score: {fairnessPercent.toFixed(0)}% • {findings.length} finding{findings.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Severity summary badges */}
          <div className="hidden sm:flex items-center gap-2">
            {getSeverityCount('critical') > 0 && (
              <span className="badge bg-red-500/20 text-red-400 border-red-500/30">
                {getSeverityCount('critical')} Critical
              </span>
            )}
            {getSeverityCount('high') > 0 && (
              <span className="badge bg-orange-500/20 text-orange-400 border-orange-500/30">
                {getSeverityCount('high')} High
              </span>
            )}
            {getSeverityCount('medium') > 0 && (
              <span className="badge bg-amber-500/20 text-amber-400 border-amber-500/30">
                {getSeverityCount('medium')} Medium
              </span>
            )}
          </div>
          
          <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-surface-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-surface-400" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-6 pb-6 space-y-6">
          {/* Stats Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-surface-800">
            {/* Circular Progress */}
            <div className="glass-panel flex flex-col items-center justify-center py-6">
              <CircularProgress 
                value={fairnessPercent} 
                size={140} 
                strokeWidth={10}
                label="Fairness"
              />
              <p className="text-sm text-surface-400 mt-4 text-center">
                Overall Fairness Score
              </p>
            </div>

            {/* Quick Stats */}
            <div className="glass-panel p-6 flex flex-col justify-center">
              <div className="flex items-center gap-3 mb-4">
                <Eye className="w-5 h-5 text-indigo-400" />
                <span className="text-sm font-medium text-surface-300">Audit Summary</span>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-surface-400">Total Findings</span>
                  <span className="text-lg font-semibold text-white">{findings.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-surface-400">Critical Issues</span>
                  <span className={`text-lg font-semibold ${getSeverityCount('critical') > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {getSeverityCount('critical')}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-surface-400">Status</span>
                  <span className={`text-sm font-semibold ${audit_passed ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {audit_passed ? 'Passed' : 'Needs Review'}
                  </span>
                </div>
              </div>
            </div>

            {/* Compliance Info */}
            <div className="glass-panel p-6 flex flex-col justify-center">
              <div className="flex items-center gap-3 mb-4">
                <Scale className="w-5 h-5 text-emerald-400" />
                <span className="text-sm font-medium text-surface-300">Compliance</span>
              </div>
              <div className="space-y-2">
                {compliance_notes.length > 0 ? (
                  compliance_notes.slice(0, 3).map((note, idx) => (
                    <div key={idx} className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span className="text-xs text-surface-400 line-clamp-2">{note}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-surface-500 italic">No compliance notes</p>
                )}
              </div>
            </div>
          </div>

          {/* Findings */}
          {findings.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Findings
              </h4>
              <div className="space-y-3">
                {findings.map((finding, idx) => {
                  const styles = getSeverityStyles(finding.severity)
                  const Icon = styles.icon
                  
                  return (
                    <div 
                      key={idx}
                      className={`p-4 rounded-xl border ${styles.bg} ${styles.border}`}
                    >
                      <div className="flex items-start gap-3">
                        <Icon className={`w-5 h-5 ${styles.text} flex-shrink-0 mt-0.5`} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <span className={`text-xs font-bold uppercase ${styles.text}`}>
                              {finding.severity}
                            </span>
                            {finding.category && (
                              <span className={`text-xs ${styles.text} opacity-70`}>
                                • {finding.category?.replace(/_/g, ' ')}
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-surface-200">{finding.description}</p>
                          {finding.recommendation && (
                            <div className="mt-2 flex items-start gap-2">
                              <TrendingUp className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
                              <p className="text-xs text-surface-400">
                                {finding.recommendation}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                Recommendations
              </h4>
              <div className="glass-panel p-4">
                <ul className="space-y-3">
                  {recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded-full bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
                        <span className="text-xs font-bold text-indigo-400">{idx + 1}</span>
                      </div>
                      <span className="text-sm text-surface-300">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Empty State for Findings */}
          {findings.length === 0 && (
            <div className="text-center py-8">
              <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
              <p className="text-surface-300 font-medium">No bias findings detected</p>
              <p className="text-sm text-surface-500 mt-1">
                The recruitment process appears to be fair and unbiased
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default BiasAuditPanel
