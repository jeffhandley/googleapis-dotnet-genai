/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Text.Json;

using Google.GenAI;
using Google.GenAI.Types;

namespace Microsoft.Extensions.AI;

/// <summary>
/// Provides an <see cref="IRealtimeSession"/> implementation for Google GenAI's Live API,
/// wrapping an <see cref="AsyncSession"/> WebSocket connection.
/// </summary>
#pragma warning disable MEAI001 // Experimental AI API
public sealed class GoogleGenAIRealtimeSession : IRealtimeSession
{
  private readonly AsyncSession _asyncSession;
  private readonly Client _client;
  private readonly string _model;
  private readonly ChatClientMetadata _metadata;
  private int _disposed;

  // Buffer for audio chunks between Append and Commit
  private readonly List<byte[]> _audioBuffer = new();
  private readonly object _audioBufferLock = new();

  // Track whether a response is in progress to emit ResponseCreated only once per response
  private bool _responseInProgress;

  // Track whether audio was sent via SendRealtimeInputAsync to avoid mixing with SendClientContentAsync
  private bool _lastInputWasRealtime;

  /// <inheritdoc />
  public RealtimeSessionOptions? Options { get; private set; }

  /// <summary>Initializes a new instance wrapping a connected <see cref="AsyncSession"/>.</summary>
  internal GoogleGenAIRealtimeSession(
    AsyncSession asyncSession,
    Client client,
    string model,
    RealtimeSessionOptions? initialOptions)
  {
    _asyncSession = asyncSession ?? throw new ArgumentNullException(nameof(asyncSession));
    _client = client ?? throw new ArgumentNullException(nameof(client));
    _model = model;
    _metadata = new ChatClientMetadata("google-genai", defaultModelId: model);
    Options = initialOptions;
  }

  /// <inheritdoc />
  /// <remarks>
  /// Google's Live API configures the session entirely at connection time via <c>ConnectAsync</c>.
  /// Mid-session reconfiguration is not supported by the protocol. This method stores the options
  /// locally for reference but does not apply them to the active session.
  /// </remarks>
  public Task UpdateAsync(RealtimeSessionOptions options, CancellationToken cancellationToken = default)
  {
    Options = options ?? throw new ArgumentNullException(nameof(options));
    return Task.CompletedTask;
  }

  /// <inheritdoc />
  public async Task SendClientMessageAsync(
    RealtimeClientMessage message,
    CancellationToken cancellationToken = default)
  {
    if (message == null) throw new ArgumentNullException(nameof(message));
    cancellationToken.ThrowIfCancellationRequested();

    try
    {
      switch (message)
      {
        case RealtimeClientInputAudioBufferAppendMessage audioAppend:
          await HandleAudioAppendAsync(audioAppend, cancellationToken).ConfigureAwait(false);
          break;

        case RealtimeClientInputAudioBufferCommitMessage:
          await HandleAudioCommitAsync(cancellationToken).ConfigureAwait(false);
          break;

        case RealtimeClientConversationItemCreateMessage itemCreate:
          await HandleConversationItemCreateAsync(itemCreate, cancellationToken).ConfigureAwait(false);
          break;

        case RealtimeClientResponseCreateMessage:
          // Google's Live API generates responses automatically after ActivityEnd
          // when using SendRealtimeInputAsync. Only send TurnComplete via
          // SendClientContentAsync when text content was sent (non-realtime path).
          // Mixing the two APIs causes unexpected behavior per Google's docs.
          if (!_lastInputWasRealtime)
          {
            await _asyncSession.SendClientContentAsync(
              new LiveSendClientContentParameters { TurnComplete = true },
              cancellationToken).ConfigureAwait(false);
          }
          break;

        default:
          // Attempt raw passthrough for unknown message types
          break;
      }
    }
    catch (Exception ex) when (ex is OperationCanceledException or ObjectDisposedException or WebSocketException)
    {
      // Expected during session teardown
    }
  }

  /// <inheritdoc />
  public async IAsyncEnumerable<RealtimeServerMessage> GetStreamingResponseAsync(
    [EnumeratorCancellation] CancellationToken cancellationToken = default)
  {
    while (!cancellationToken.IsCancellationRequested)
    {
      LiveServerMessage? serverMessage;
      try
      {
        serverMessage = await _asyncSession.ReceiveAsync(cancellationToken).ConfigureAwait(false);
      }
      catch (Exception ex) when (ex is OperationCanceledException or ObjectDisposedException or WebSocketException)
      {
        yield break;
      }

      if (serverMessage is null)
      {
        yield break;
      }

      // Map Google Live server messages to MEAI server message types
      foreach (var mapped in MapServerMessage(serverMessage))
      {
        yield return mapped;
      }
    }
  }

  /// <inheritdoc />
  public object? GetService(System.Type serviceType, object? serviceKey = null)
  {
    if (serviceType == null) throw new ArgumentNullException(nameof(serviceType));

    if (serviceKey is not null) return null;

    if (serviceType == typeof(ChatClientMetadata)) return _metadata;
    if (serviceType.IsInstanceOfType(this)) return this;
    if (serviceType.IsInstanceOfType(_asyncSession)) return _asyncSession;

    return null;
  }

  /// <inheritdoc />
  public void Dispose()
  {
    if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

    // Fire-and-forget async disposal on a thread pool thread to avoid
    // deadlocking when called from a UI thread with a SynchronizationContext.
    _ = Task.Run(async () =>
    {
      try
      {
        await _asyncSession.DisposeAsync().ConfigureAwait(false);
      }
      catch (Exception ex) when (ex is ObjectDisposedException or WebSocketException)
      {
        // Already disposed or disconnected
      }
    });
  }

  /// <inheritdoc />
  public async ValueTask DisposeAsync()
  {
    if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
    await _asyncSession.DisposeAsync().ConfigureAwait(false);
  }

  #region Send Helpers (MEAI → Google GenAI)

  private Task HandleAudioAppendAsync(
    RealtimeClientInputAudioBufferAppendMessage audioAppend,
    CancellationToken cancellationToken)
  {
    if (audioAppend.Content is null || !audioAppend.Content.HasTopLevelMediaType("audio"))
    {
      return Task.CompletedTask;
    }

    byte[] audioBytes = ExtractAudioBytes(audioAppend.Content);

    // Send audio directly as realtime input (Google uses VAD)
    _lastInputWasRealtime = true;
    return _asyncSession.SendRealtimeInputAsync(
      new LiveSendRealtimeInputParameters
      {
        Audio = new Blob
        {
          Data = audioBytes,
          MimeType = "audio/pcm",
        }
      },
      cancellationToken);
  }

  private async Task HandleAudioCommitAsync(CancellationToken cancellationToken)
  {
    // Signal end of audio activity to trigger response generation
    await _asyncSession.SendRealtimeInputAsync(
      new LiveSendRealtimeInputParameters
      {
        ActivityEnd = new ActivityEnd()
      },
      cancellationToken).ConfigureAwait(false);
  }

  private async Task HandleConversationItemCreateAsync(
    RealtimeClientConversationItemCreateMessage itemCreate,
    CancellationToken cancellationToken)
  {
    if (itemCreate.Item?.Contents is null or { Count: 0 })
    {
      return;
    }

    // Check if this is a function result
    var firstContent = itemCreate.Item.Contents[0];
    if (firstContent is FunctionResultContent functionResult)
    {
      var response = new FunctionResponse
      {
        Id = functionResult.CallId,
        Name = string.Empty,
        Response = new Dictionary<string, object>
        {
          ["result"] = functionResult.Result?.ToString() ?? string.Empty
        }
      };

      await _asyncSession.SendToolResponseAsync(
        new LiveSendToolResponseParameters
        {
          FunctionResponses = new List<FunctionResponse> { response }
        },
        cancellationToken).ConfigureAwait(false);
      return;
    }

    // Otherwise, treat as text/content conversation input
    var parts = new List<Part>();
    foreach (var content in itemCreate.Item.Contents)
    {
      if (content is TextContent textContent && !string.IsNullOrEmpty(textContent.Text))
      {
        parts.Add(new Part { Text = textContent.Text });
      }
      else if (content is DataContent dataContent)
      {
        if (dataContent.HasTopLevelMediaType("audio"))
        {
          parts.Add(new Part
          {
            InlineData = new Blob
            {
              Data = ExtractAudioBytes(dataContent),
              MimeType = dataContent.MediaType ?? "audio/pcm",
            }
          });
        }
        else if (dataContent.HasTopLevelMediaType("image") && dataContent.Uri is not null)
        {
          parts.Add(new Part
          {
            FileData = new FileData { FileUri = dataContent.Uri }
          });
        }
      }
    }

    if (parts.Count == 0) return;

    string role = itemCreate.Item.Role?.Value switch
    {
      "assistant" => "model",
      _ => "user",
    };

    _lastInputWasRealtime = false;
    await _asyncSession.SendClientContentAsync(
      new LiveSendClientContentParameters
      {
        Turns = new List<Content>
        {
          new Content
          {
            Parts = parts,
            Role = role,
          }
        },
        TurnComplete = true,
      },
      cancellationToken).ConfigureAwait(false);
  }

  private static byte[] ExtractAudioBytes(DataContent content)
  {
    string? dataUri = content.Uri?.ToString();

    if (dataUri is not null)
    {
      int commaIndex = dataUri.LastIndexOf(',');
      if (commaIndex >= 0 && commaIndex < dataUri.Length - 1)
      {
        string base64 = dataUri.Substring(commaIndex + 1);
        return Convert.FromBase64String(base64);
      }
    }

    return content.Data.ToArray();
  }

  #endregion

  #region Receive Helpers (Google GenAI → MEAI)

  private IEnumerable<RealtimeServerMessage> MapServerMessage(LiveServerMessage serverMessage)
  {
    // SetupComplete — skip (internal protocol message, not relevant to MEAI consumers)
    if (serverMessage.SetupComplete is not null)
    {
      yield break;
    }

    // Server content (model responses — audio, text, transcription)
    if (serverMessage.ServerContent is { } serverContent)
    {
      foreach (var msg in MapServerContent(serverContent, serverMessage))
      {
        yield return msg;
      }
    }

    // Tool calls — emit ResponseCreated (if not already), then ResponseOutputItemAdded + ResponseOutputItemDone for each
    if (serverMessage.ToolCall is { FunctionCalls: { Count: > 0 } functionCalls })
    {
      if (!_responseInProgress)
      {
        _responseInProgress = true;
        yield return new RealtimeServerResponseCreatedMessage(RealtimeServerMessageType.ResponseCreated)
        {
          RawRepresentation = serverMessage,
        };
      }

      foreach (var fc in functionCalls)
      {
        var contents = new List<AIContent>
        {
          new FunctionCallContent(fc.Id ?? string.Empty, fc.Name ?? string.Empty, fc.Args?.ToDictionary(kvp => kvp.Key, kvp => (object?)kvp.Value))
        };

        var item = new RealtimeContentItem(contents, id: fc.Id, role: ChatRole.Assistant);

        // Emit ResponseOutputItemAdded (signals start of output item)
        yield return new RealtimeServerResponseOutputItemMessage(RealtimeServerMessageType.ResponseOutputItemAdded)
        {
          Item = item,
          RawRepresentation = serverMessage,
        };

        // Emit ResponseOutputItemDone (required by FunctionInvokingRealtimeSession middleware)
        yield return new RealtimeServerResponseOutputItemMessage(RealtimeServerMessageType.ResponseOutputItemDone)
        {
          Item = item,
          RawRepresentation = serverMessage,
        };
      }
    }

    // Tool call cancellation
    if (serverMessage.ToolCallCancellation is { Ids: { Count: > 0 } })
    {
      yield return new RealtimeServerMessage
      {
        Type = RealtimeServerMessageType.RawContentOnly,
        RawRepresentation = serverMessage,
      };
    }

    // Usage metadata
    if (serverMessage.UsageMetadata is { } usage)
    {
      yield return new RealtimeServerResponseCreatedMessage(RealtimeServerMessageType.ResponseDone)
      {
        Usage = new UsageDetails
        {
          InputTokenCount = usage.PromptTokenCount ?? 0,
          OutputTokenCount = usage.ResponseTokenCount ?? 0,
          TotalTokenCount = usage.TotalTokenCount ?? 0,
        },
        RawRepresentation = serverMessage,
      };
    }

    // GoAway (server disconnect)
    if (serverMessage.GoAway is not null)
    {
      yield return new RealtimeServerErrorMessage
      {
        Error = new ErrorContent("Server is disconnecting (GoAway)"),
        RawRepresentation = serverMessage,
      };
    }
  }

  private IEnumerable<RealtimeServerMessage> MapServerContent(
    LiveServerContent serverContent,
    LiveServerMessage rawMessage)
  {
    if (serverContent.ModelTurn?.Parts is { Count: > 0 } parts)
    {
      // Emit ResponseCreated once when a new response cycle begins
      if (!_responseInProgress)
      {
        _responseInProgress = true;
        yield return new RealtimeServerResponseCreatedMessage(RealtimeServerMessageType.ResponseCreated)
        {
          RawRepresentation = rawMessage,
        };
      }

      foreach (var part in parts)
      {
        // Audio data
        if (part.InlineData is { Data: not null } blob &&
            blob.MimeType?.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) == true)
        {
          yield return new RealtimeServerOutputTextAudioMessage(RealtimeServerMessageType.OutputAudioDelta)
          {
            Audio = Convert.ToBase64String(blob.Data),
            RawRepresentation = rawMessage,
          };
        }

        // Text response
        if (!string.IsNullOrEmpty(part.Text))
        {
          yield return new RealtimeServerOutputTextAudioMessage(RealtimeServerMessageType.OutputTextDelta)
          {
            Text = part.Text,
            RawRepresentation = rawMessage,
          };
        }
      }
    }

    // Input transcription
    if (serverContent.InputTranscription is { Text: not null } inputTranscription)
    {
      yield return new RealtimeServerInputAudioTranscriptionMessage(RealtimeServerMessageType.InputAudioTranscriptionCompleted)
      {
        Transcription = inputTranscription.Text,
        RawRepresentation = rawMessage,
      };
    }

    // Output transcription
    if (serverContent.OutputTranscription is { Text: not null } outputTranscription)
    {
      yield return new RealtimeServerOutputTextAudioMessage(RealtimeServerMessageType.OutputAudioTranscriptionDelta)
      {
        Text = outputTranscription.Text,
        RawRepresentation = rawMessage,
      };
    }

    // Turn complete or generation complete — reset response tracking and emit ResponseDone
    if (serverContent.TurnComplete == true || serverContent.GenerationComplete == true)
    {
      _responseInProgress = false;
      yield return new RealtimeServerResponseCreatedMessage(RealtimeServerMessageType.ResponseDone)
      {
        RawRepresentation = rawMessage,
      };
    }
  }

  #endregion

  #region Tool Mapping Helpers

  /// <summary>
  /// Converts an <see cref="AIFunction"/> to a Google GenAI <see cref="FunctionDeclaration"/>,
  /// mapping the function name, description, and JSON schema for parameters.
  /// </summary>
  /// <param name="aiFunction">The AI function to convert.</param>
  /// <returns>A Google GenAI function declaration.</returns>
  internal static FunctionDeclaration ToGoogleFunctionDeclaration(AIFunction aiFunction)
  {
    var declaration = new FunctionDeclaration
    {
      Name = aiFunction.Name,
      Description = aiFunction.Description,
    };

    // Map the JSON schema for parameters
    if (aiFunction.JsonSchema is JsonElement schemaElement &&
        schemaElement.ValueKind != JsonValueKind.Undefined)
    {
      declaration.ParametersJsonSchema = schemaElement;
    }

    return declaration;
  }

  #endregion
}
#pragma warning restore MEAI001
