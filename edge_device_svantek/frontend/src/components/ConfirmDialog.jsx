export default function ConfirmDialog({ title, message, onConfirm, onCancel }) {
  return (
    <div className="dialog-backdrop" onClick={onCancel}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h3 className="dialog-title">{title}</h3>
        <p className="dialog-message">{message}</p>
        <div className="dialog-actions">
          <button className="dialog-cancel" onClick={onCancel}>
            Vazgeç
          </button>
          <button className="dialog-confirm" onClick={onConfirm}>
            Sil
          </button>
        </div>
      </div>
    </div>
  );
}
