import { useState, useEffect, useCallback } from 'react'
import { 
  Clock, 
  CheckCircle, 
  ChevronRight, 
  ChevronLeft,
  AlertCircle,
  Trophy,
  Brain,
  Loader2
} from 'lucide-react'

function TestingCenter({ 
  questions = [], 
  candidateId, 
  pipelineId,
  onSubmit, 
  onClose,
  isSubmitting = false 
}) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [answers, setAnswers] = useState({})
  const [timeSpent, setTimeSpent] = useState({})
  const [questionStartTime, setQuestionStartTime] = useState(Date.now())
  const [totalTime, setTotalTime] = useState(0)
  const [showResults, setShowResults] = useState(false)

  const currentQuestion = questions[currentIndex]
  const isLastQuestion = currentIndex === questions.length - 1
  const answeredCount = Object.keys(answers).length
  const progress = (answeredCount / questions.length) * 100

  // Track total time
  useEffect(() => {
    const interval = setInterval(() => {
      setTotalTime((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  // Reset question timer when moving to new question
  useEffect(() => {
    setQuestionStartTime(Date.now())
  }, [currentIndex])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleSelectOption = (option) => {
    const questionId = currentQuestion?.question_id || currentIndex
    const timeOnQuestion = Math.round((Date.now() - questionStartTime) / 1000)
    
    setAnswers((prev) => ({ ...prev, [questionId]: option }))
    setTimeSpent((prev) => ({ ...prev, [questionId]: timeOnQuestion }))
  }

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex((prev) => prev + 1)
    }
  }

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1)
    }
  }

  const handleSubmit = async () => {
    // Format responses for API
    const responses = questions.map((q, idx) => {
      const questionId = q.question_id || idx
      return {
        question_id: questionId,
        selected_option: answers[questionId] || null,
        time_seconds: timeSpent[questionId] || 0,
      }
    })

    await onSubmit({
      candidate_id: candidateId,
      pipeline_id: pipelineId,
      responses,
      total_time_seconds: totalTime,
    })
  }

  const getDifficultyColor = (difficulty) => {
    switch (difficulty?.toLowerCase()) {
      case 'easy': return 'text-emerald-400 bg-emerald-500/20'
      case 'medium': return 'text-amber-400 bg-amber-500/20'
      case 'hard': return 'text-red-400 bg-red-500/20'
      default: return 'text-surface-400 bg-surface-700'
    }
  }

  if (!questions.length) {
    return (
      <div className="glass-card p-12 text-center">
        <Brain className="w-16 h-16 text-surface-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-white mb-2">No Questions Available</h3>
        <p className="text-surface-400 mb-6">
          There are no test questions to display at this time.
        </p>
        <button onClick={onClose} className="btn-secondary">
          Go Back
        </button>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-surface-950 p-6">
      {/* Header */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-400 flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">Technical Assessment</h1>
                <p className="text-sm text-surface-400">
                  Question {currentIndex + 1} of {questions.length}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              {/* Timer */}
              <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-800 border border-surface-700">
                <Clock className="w-5 h-5 text-indigo-400" />
                <span className="text-lg font-mono font-semibold text-white">
                  {formatTime(totalTime)}
                </span>
              </div>

              {/* Answered Count */}
              <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-800 border border-surface-700">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                <span className="text-lg font-semibold text-white">
                  {answeredCount}/{questions.length}
                </span>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-indigo-600 to-emerald-500 transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Question Card */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="glass-card p-8">
          {/* Question Header */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3">
              <span className="text-3xl font-bold text-indigo-400">
                Q{currentIndex + 1}
              </span>
              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getDifficultyColor(currentQuestion?.difficulty)}`}>
                {currentQuestion?.difficulty || 'Medium'}
              </span>
              {currentQuestion?.topic && (
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-surface-800 text-surface-300">
                  {currentQuestion.topic}
                </span>
              )}
            </div>
            
            {currentQuestion?.time_limit_seconds && (
              <div className="text-sm text-surface-500">
                Suggested time: {Math.round(currentQuestion.time_limit_seconds / 60)} min
              </div>
            )}
          </div>

          {/* Question Text */}
          <div className="mb-8">
            <p className="text-lg text-white leading-relaxed">
              {currentQuestion?.question_text}
            </p>
          </div>

          {/* Options */}
          <div className="space-y-3">
            {currentQuestion?.options && Object.entries(currentQuestion.options).map(([key, value]) => {
              const questionId = currentQuestion?.question_id || currentIndex
              const isSelected = answers[questionId] === key

              return (
                <button
                  key={key}
                  onClick={() => handleSelectOption(key)}
                  className={`
                    w-full flex items-center gap-4 p-4 rounded-xl
                    border-2 transition-all duration-200
                    text-left group
                    ${isSelected
                      ? 'border-indigo-500 bg-indigo-500/10 shadow-lg shadow-indigo-500/20'
                      : 'border-surface-700 bg-surface-800/50 hover:border-surface-500 hover:bg-surface-800'
                    }
                  `}
                >
                  <div className={`
                    w-10 h-10 rounded-lg flex items-center justify-center font-bold text-lg
                    transition-all duration-200
                    ${isSelected
                      ? 'bg-indigo-500 text-white'
                      : 'bg-surface-700 text-surface-400 group-hover:bg-surface-600'
                    }
                  `}>
                    {key}
                  </div>
                  <span className={`flex-1 ${isSelected ? 'text-white' : 'text-surface-300'}`}>
                    {value}
                  </span>
                  {isSelected && (
                    <CheckCircle className="w-6 h-6 text-indigo-400" />
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="max-w-4xl mx-auto">
        <div className="glass-card p-6">
          <div className="flex items-center justify-between">
            <button
              onClick={handlePrevious}
              disabled={currentIndex === 0}
              className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-5 h-5" />
              Previous
            </button>

            {/* Question Dots */}
            <div className="flex items-center gap-2 flex-wrap justify-center max-w-lg">
              {questions.map((q, idx) => {
                const questionId = q.question_id || idx
                const isAnswered = answers[questionId] !== undefined
                const isCurrent = idx === currentIndex

                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentIndex(idx)}
                    className={`
                      w-8 h-8 rounded-lg text-sm font-semibold
                      transition-all duration-200
                      ${isCurrent
                        ? 'bg-indigo-500 text-white scale-110'
                        : isAnswered
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-surface-800 text-surface-400 hover:bg-surface-700'
                      }
                    `}
                  >
                    {idx + 1}
                  </button>
                )
              })}
            </div>

            {isLastQuestion ? (
              <button
                onClick={handleSubmit}
                disabled={isSubmitting || answeredCount < questions.length}
                className="btn-success disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  <>
                    <Trophy className="w-5 h-5" />
                    Submit Test
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={handleNext}
                className="btn-primary"
              >
                Next
                <ChevronRight className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Warning if not all answered */}
          {answeredCount < questions.length && isLastQuestion && (
            <div className="mt-4 flex items-center gap-2 justify-center text-amber-400 text-sm">
              <AlertCircle className="w-4 h-4" />
              <span>Please answer all questions before submitting</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TestingCenter
