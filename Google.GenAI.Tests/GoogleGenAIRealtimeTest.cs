using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Google.GenAI.Types;
using Microsoft.Extensions.AI;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Google.GenAI.Tests;

[TestClass]
public class GoogleGenAIRealtimeExtensionsTest
{
  [TestMethod]
  public void AsIRealtimeClient_NullClient_ThrowsArgumentNullException()
  {
    Client? client = null;
    Assert.ThrowsException<ArgumentNullException>(() => client!.AsIRealtimeClient("model"));
  }

  [TestMethod]
  public void AsIRealtimeClient_ValidClient_ReturnsNonNull()
  {
    var client = new Client(apiKey: "fake-api-key");
    var realtimeClient = client.AsIRealtimeClient("gemini-2.5-flash-native-audio-preview-12-2025");

    Assert.IsNotNull(realtimeClient);
  }

  [TestMethod]
  public void AsIRealtimeClient_NullModelId_Succeeds()
  {
    var client = new Client(apiKey: "fake-api-key");
    var realtimeClient = client.AsIRealtimeClient();

    Assert.IsNotNull(realtimeClient);
  }
}

[TestClass]
public class GoogleGenAIRealtimeClientTest
{
  [TestMethod]
  public void Ctor_NullClient_Throws()
  {
    Assert.ThrowsException<ArgumentNullException>(() =>
      new GoogleGenAIRealtimeClient(null!, "model"));
  }

  [TestMethod]
  public void Ctor_NullModelId_Succeeds()
  {
    var client = new Client(apiKey: "fake-api-key");
    var realtimeClient = new GoogleGenAIRealtimeClient(client, defaultModelId: null);
    Assert.IsNotNull(realtimeClient);
  }

  [TestMethod]
  public void GetService_ReturnsExpectedServices()
  {
    var client = new Client(apiKey: "fake-api-key");
    using IRealtimeClient realtimeClient = new GoogleGenAIRealtimeClient(client, "model");

    Assert.AreSame(realtimeClient, realtimeClient.GetService(typeof(GoogleGenAIRealtimeClient)));
    Assert.AreSame(realtimeClient, realtimeClient.GetService(typeof(IRealtimeClient)));
    Assert.AreSame(client, realtimeClient.GetService(typeof(Client)));
    Assert.IsNull(realtimeClient.GetService(typeof(string)));
    Assert.IsNull(realtimeClient.GetService(typeof(GoogleGenAIRealtimeClient), "someKey"));
  }

  [TestMethod]
  public void GetService_NullServiceType_Throws()
  {
    var client = new Client(apiKey: "fake-api-key");
    using IRealtimeClient realtimeClient = new GoogleGenAIRealtimeClient(client, "model");

    Assert.ThrowsException<ArgumentNullException>(() => realtimeClient.GetService(null!));
  }

  [TestMethod]
  public void GetService_ChatClientMetadata_ReturnsCorrectMetadata()
  {
    var client = new Client(apiKey: "fake-api-key");
    using IRealtimeClient realtimeClient = new GoogleGenAIRealtimeClient(client, "test-model");

    var metadata = realtimeClient.GetService(typeof(ChatClientMetadata)) as ChatClientMetadata;

    Assert.IsNotNull(metadata);
    Assert.AreEqual("google-genai", metadata.ProviderName);
    Assert.AreEqual("test-model", metadata.DefaultModelId);
  }

  [TestMethod]
  public void GetService_WithServiceKey_ReturnsNull()
  {
    var client = new Client(apiKey: "fake-api-key");
    using IRealtimeClient realtimeClient = new GoogleGenAIRealtimeClient(client, "model");

    Assert.IsNull(realtimeClient.GetService(typeof(ChatClientMetadata), "key"));
    Assert.IsNull(realtimeClient.GetService(typeof(GoogleGenAIRealtimeClient), "key"));
  }

  [TestMethod]
  public void Dispose_CanBeCalledMultipleTimes()
  {
    var client = new Client(apiKey: "fake-api-key");
    IRealtimeClient realtimeClient = new GoogleGenAIRealtimeClient(client, "model");
    realtimeClient.Dispose();

    // Second dispose should not throw
    realtimeClient.Dispose();
  }

  [TestMethod]
  public async Task CreateSessionAsync_NoModelAnywhere_Throws()
  {
    var client = new Client(apiKey: "fake-api-key");
    using var realtimeClient = new GoogleGenAIRealtimeClient(client, defaultModelId: null);

    await Assert.ThrowsExceptionAsync<InvalidOperationException>(
      () => realtimeClient.CreateSessionAsync());
  }

  [TestMethod]
  public async Task CreateSessionAsync_ModelFromOptions_UsesOptionsModel()
  {
    var client = new Client(apiKey: "fake-api-key");
    using var realtimeClient = new GoogleGenAIRealtimeClient(client, defaultModelId: null);

    var options = new RealtimeSessionOptions { Model = "gemini-2.5-flash-native-audio-preview-12-2025" };

    // This will attempt to actually connect and fail, but we're testing model resolution
    // not actual connectivity. The InvalidOperationException from missing model should NOT be thrown.
    try
    {
      await realtimeClient.CreateSessionAsync(options);
    }
    catch (Exception ex) when (ex is not InvalidOperationException)
    {
      // Expected — connection failure is fine, we just want to verify no "No model specified" error
    }
  }

  [TestMethod]
  public async Task CreateSessionAsync_Cancelled_Throws()
  {
    var client = new Client(apiKey: "fake-api-key");
    using var realtimeClient = new GoogleGenAIRealtimeClient(client, "model");
    using var cts = new CancellationTokenSource();
    cts.Cancel();

    try
    {
      await realtimeClient.CreateSessionAsync(cancellationToken: cts.Token);
      Assert.Fail("Expected OperationCanceledException was not thrown.");
    }
    catch (OperationCanceledException)
    {
      // Expected — TaskCanceledException is a subclass of OperationCanceledException
    }
  }

  [TestMethod]
  public void Dispose_AfterDispose_GetServiceStillWorks()
  {
    var client = new Client(apiKey: "fake-api-key");
    IRealtimeClient realtimeClient = new GoogleGenAIRealtimeClient(client, "model");
    realtimeClient.Dispose();
    realtimeClient.Dispose();

    // GetService should still work after dispose (same pattern as OpenAI)
    Assert.IsNull(realtimeClient.GetService(typeof(string)));
  }
}

[TestClass]
public class GoogleGenAIRealtimeSessionTest
{
  // GoogleGenAIRealtimeSession requires a connected AsyncSession which requires real connectivity.
  // These tests validate the static/helper methods and type behavior that can be tested without
  // a live WebSocket connection.

  [TestMethod]
  public void ToGoogleFunctionDeclaration_MapsNameAndDescription()
  {
    var schemaJson = """{"type":"object","properties":{"city":{"type":"string"}}}""";
    var schemaElement = JsonDocument.Parse(schemaJson).RootElement;

    var aiFunction = new TestAIFunction("get_weather", "Gets the weather", schemaElement);

    var declaration = GoogleGenAIRealtimeSession.ToGoogleFunctionDeclaration(aiFunction);

    Assert.AreEqual("get_weather", declaration.Name);
    Assert.AreEqual("Gets the weather", declaration.Description);
  }

  [TestMethod]
  public void ToGoogleFunctionDeclaration_MapsJsonSchema()
  {
    var schemaJson = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""";
    var schemaElement = JsonDocument.Parse(schemaJson).RootElement;

    var aiFunction = new TestAIFunction("test_func", "desc", schemaElement);

    var declaration = GoogleGenAIRealtimeSession.ToGoogleFunctionDeclaration(aiFunction);

    Assert.IsNotNull(declaration.ParametersJsonSchema);
  }

  [TestMethod]
  public void ToGoogleFunctionDeclaration_DefaultSchema_SetsNoParameters()
  {
    var aiFunction = new TestAIFunction("simple", "A simple function", default);

    var declaration = GoogleGenAIRealtimeSession.ToGoogleFunctionDeclaration(aiFunction);

    Assert.AreEqual("simple", declaration.Name);
    Assert.IsNull(declaration.ParametersJsonSchema);
  }

  /// <summary>Test implementation of AIFunction for unit testing.</summary>
  private sealed class TestAIFunction : AIFunction
  {
    public TestAIFunction(string name, string description, JsonElement jsonSchema)
    {
      Name = name;
      Description = description;
      JsonSchema = jsonSchema;
    }

    public override string Name { get; }
    public override string Description { get; }
    public override JsonElement JsonSchema { get; }

    protected override ValueTask<object?> InvokeCoreAsync(
      AIFunctionArguments arguments,
      CancellationToken cancellationToken) => new((object?)null);
  }
}
