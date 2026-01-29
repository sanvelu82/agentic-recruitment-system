import { useState, useRef, useCallback } from 'react'
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { candidatesApi } from '../services/api'

/**
 * PDF Resume Upload Component
 * 
 * Features:
 * - Drag and drop support
 * - File type validation
 * - Upload progress indicator
 * - Error handling with user feedback
 */
export default function ResumeUploader({ jobId, onUploadSuccess, onUploadError, disabled }) {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadResult, setUploadResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  // Validate file before upload
  const validateFile = (file) => {
    const maxSize = 10 * 1024 * 1024 // 10MB
    const allowedTypes = ['application/pdf']
    
    if (!file) {
      return 'No file selected'
    }
    
    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.pdf')) {
      return 'Only PDF files are allowed'
    }
    
    if (file.size > maxSize) {
      return `File too large. Maximum size is 10MB (your file: ${(file.size / 1024 / 1024).toFixed(1)}MB)`
    }
    
    return null
  }

  // Handle file selection
  const handleFileSelect = useCallback((selectedFile) => {
    setError(null)
    setUploadResult(null)
    
    const validationError = validateFile(selectedFile)
    if (validationError) {
      setError(validationError)
      setFile(null)
      return
    }
    
    setFile(selectedFile)
  }, [])

  // Handle drag events
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled && !uploading) {
      setIsDragging(true)
    }
  }, [disabled, uploading])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    
    if (disabled || uploading) return
    
    const droppedFile = e.dataTransfer.files[0]
    handleFileSelect(droppedFile)
  }, [disabled, uploading, handleFileSelect])

  // Handle file input change
  const handleInputChange = (e) => {
    const selectedFile = e.target.files[0]
    handleFileSelect(selectedFile)
  }

  // Handle upload
  const handleUpload = async () => {
    if (!file || !jobId || uploading) return
    
    setUploading(true)
    setUploadProgress(0)
    setError(null)
    setUploadResult(null)
    
    try {
      const result = await candidatesApi.uploadResume(
        jobId,
        file,
        'pdf_upload',
        (progress) => setUploadProgress(progress)
      )
      
      setUploadResult(result)
      setFile(null)
      setUploadProgress(100)
      
      if (onUploadSuccess) {
        onUploadSuccess(result)
      }
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Upload failed'
      setError(errorMessage)
      
      if (onUploadError) {
        onUploadError(errorMessage)
      }
    } finally {
      setUploading(false)
    }
  }

  // Reset state
  const handleReset = () => {
    setFile(null)
    setError(null)
    setUploadResult(null)
    setUploadProgress(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="w-full">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && !uploading && fileInputRef.current?.click()}
        className={`
          relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
          transition-all duration-200
          ${isDragging 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
            : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
          }
          ${disabled || uploading ? 'opacity-50 cursor-not-allowed' : ''}
          ${error ? 'border-red-300 dark:border-red-600' : ''}
          ${uploadResult ? 'border-green-300 dark:border-green-600' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleInputChange}
          className="hidden"
          disabled={disabled || uploading}
        />
        
        {/* Icon and Text */}
        <div className="flex flex-col items-center gap-2">
          {uploading ? (
            <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
          ) : uploadResult ? (
            <CheckCircle className="w-10 h-10 text-green-500" />
          ) : error ? (
            <AlertCircle className="w-10 h-10 text-red-500" />
          ) : file ? (
            <FileText className="w-10 h-10 text-blue-500" />
          ) : (
            <Upload className="w-10 h-10 text-gray-400" />
          )}
          
          {uploading ? (
            <div className="w-full max-w-xs">
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                Uploading... {uploadProgress}%
              </p>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          ) : uploadResult ? (
            <div className="text-center">
              <p className="text-sm font-medium text-green-600 dark:text-green-400">
                Resume uploaded successfully!
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {uploadResult.pages_processed} pages • {uploadResult.characters_extracted.toLocaleString()} characters
              </p>
            </div>
          ) : error ? (
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          ) : file ? (
            <div className="text-center">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-200">
                {file.name}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div className="text-center">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-200">
                Drag and drop your PDF resume here
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                or click to browse (max 10MB)
              </p>
            </div>
          )}
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="flex gap-2 mt-3">
        {file && !uploading && !uploadResult && (
          <>
            <button
              onClick={handleUpload}
              disabled={disabled}
              className="flex-1 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg 
                hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                transition-colors duration-200 flex items-center justify-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Upload Resume
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 
                text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600
                transition-colors duration-200"
            >
              <X className="w-4 h-4" />
            </button>
          </>
        )}
        
        {(uploadResult || error) && (
          <button
            onClick={handleReset}
            className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 
              text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600
              transition-colors duration-200"
          >
            Upload Another Resume
          </button>
        )}
      </div>
      
      {/* Warnings from upload */}
      {uploadResult?.warnings?.length > 0 && (
        <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <p className="text-xs text-yellow-700 dark:text-yellow-300">
            ⚠️ {uploadResult.warnings.join('. ')}
          </p>
        </div>
      )}
    </div>
  )
}
